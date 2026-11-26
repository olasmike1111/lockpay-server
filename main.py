"""LockPay backend: customers, devices, installments, payments, signed LOCK/UNLOCK commands.

Run:
    export JWT_SECRET=$(python -c "import secrets;print(secrets.token_hex(32))")
    export ADMIN_PASSWORD='choose-a-strong-one'
    uvicorn main:app --reload
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import date, datetime, timedelta, timezone
from typing import Literal, Optional

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import ForeignKey, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# ---------- config ----------
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./lockpay.db")
# Hosts (Render etc.) hand out postgres:// or postgresql:// URLs; SQLAlchemy needs the driver named.
if DB_URL.startswith("postgres://"):
    DB_URL = "postgresql+psycopg2://" + DB_URL[len("postgres://"):]
elif DB_URL.startswith("postgresql://"):
    DB_URL = "postgresql+psycopg2://" + DB_URL[len("postgresql://"):]
JWT_SECRET = os.environ["JWT_SECRET"]
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]
KEY_PATH = os.getenv("SIGNING_KEY_PATH", "signing_key.pem")
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "")
COMMAND_TTL_HOURS = 24


def now() -> datetime:
    """Naive UTC (SQLite drops tzinfo)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# ---------- command signing (Ed25519) ----------
# Private key stays on the server. The Android app embeds only the PUBLIC key,
# so a compromised phone can't forge commands.
def load_signing_key() -> Ed25519PrivateKey:
    # Hosted: the key comes from an env var (the server's disk is wiped on restart).
    b64 = os.getenv("SIGNING_KEY_B64")
    if b64:
        return serialization.load_pem_private_key(base64.b64decode(b64), password=None)
    if os.getenv("RENDER"):
        # Never silently generate a throwaway key on a host: every phone would stop obeying us.
        raise RuntimeError("Set SIGNING_KEY_B64 (base64 of your signing_key.pem)")
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    key = Ed25519PrivateKey.generate()
    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    os.chmod(KEY_PATH, 0o600)
    return key


SIGNING_KEY = load_signing_key()
PUBLIC_KEY_B64 = base64.b64encode(
    SIGNING_KEY.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
).decode()


# ---------- database ----------
class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    phone: Mapped[str]
    address: Mapped[Optional[str]]
    created_at: Mapped[datetime] = mapped_column(default=now)


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    imei: Mapped[str] = mapped_column(unique=True)
    model: Mapped[str]
    token_hash: Mapped[str]
    lock_state: Mapped[str] = mapped_column(default="UNLOCKED")  # confirmed by device ack
    last_seen: Mapped[Optional[datetime]]
    enrolled_at: Mapped[datetime] = mapped_column(default=now)


class Installment(Base):
    __tablename__ = "installments"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    seq: Mapped[int]
    due_date: Mapped[date]
    amount: Mapped[int]          # minor units (cents/kobo/etc.)
    paid: Mapped[int] = mapped_column(default=0)


class Payment(Base):
    __tablename__ = "payments"
    id: Mapped[int] = mapped_column(primary_key=True)
    installment_id: Mapped[int] = mapped_column(ForeignKey("installments.id"))
    amount: Mapped[int]
    method: Mapped[str]
    paid_at: Mapped[datetime] = mapped_column(default=now)
    recorded_by: Mapped[str]


class Command(Base):
    __tablename__ = "commands"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id"))
    action: Mapped[str]          # LOCK | UNLOCK
    reason: Mapped[str]
    nonce: Mapped[str]
    issued_by: Mapped[str]
    issued_at: Mapped[datetime] = mapped_column(default=now)
    expires_at: Mapped[datetime]
    acked_at: Mapped[Optional[datetime]]


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(default=now)
    actor: Mapped[str]
    event: Mapped[str]
    detail: Mapped[str]


engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
Base.metadata.create_all(engine)


def get_db():
    with Session(engine) as db:
        yield db


def audit(db: Session, actor: str, event: str, detail: str = ""):
    db.add(AuditLog(actor=actor, event=event, detail=detail))


# ---------- auth ----------
def current_admin(authorization: str = Header()) -> str:
    try:
        claims = jwt.decode(authorization.removeprefix("Bearer "), JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    return claims["sub"]


def current_device(authorization: str = Header(), db: Session = Depends(get_db)) -> Device:
    token_hash = hashlib.sha256(authorization.removeprefix("Bearer ").encode()).hexdigest()
    dev = db.scalar(select(Device).where(Device.token_hash == token_hash))
    if not dev:
        raise HTTPException(401, "Unknown device")
    return dev


# ---------- schemas ----------
class LoginIn(BaseModel):
    username: str
    password: str


class CustomerIn(BaseModel):
    name: str
    phone: str
    address: Optional[str] = None


class DeviceIn(BaseModel):
    customer_id: int
    imei: str
    model: str = "Smart 9 HD"


class PlanIn(BaseModel):
    total: int = Field(gt=0)
    count: int = Field(gt=0, le=60)
    first_due: date
    interval_days: int = 30


class PaymentIn(BaseModel):
    amount: int = Field(gt=0)
    method: str = "cash"


class CommandIn(BaseModel):
    action: Literal["LOCK", "UNLOCK", "RELEASE"]
    reason: str = Field(min_length=3)


# ---------- helpers ----------
def sign_command(c: Command) -> dict:
    payload = {
        "id": c.id,
        "device_id": c.device_id,
        "action": c.action,
        "nonce": c.nonce,
        "issued_at": ts(c.issued_at),
        "expires_at": ts(c.expires_at),
    }
    msg = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"payload": payload, "signature": base64.b64encode(SIGNING_KEY.sign(msg)).decode()}


def account_summary(db: Session, customer_id: int) -> dict:
    rows = db.scalars(
        select(Installment).where(Installment.customer_id == customer_id).order_by(Installment.seq)
    ).all()
    today = date.today()
    remaining = [r for r in rows if r.paid < r.amount]
    return {
        "balance": sum(r.amount - r.paid for r in remaining),
        "overdue": sum(r.amount - r.paid for r in remaining if r.due_date < today),
        "next_due": min((r.due_date for r in remaining), default=None),
    }


# ---------- app ----------
app = FastAPI(title="LockPay")


@app.get("/public-key")
def public_key():
    """Embed this in the Android app to verify command signatures."""
    return {"alg": "Ed25519", "public_key_b64": PUBLIC_KEY_B64}


@app.post("/admin/login")
def login(body: LoginIn):
    ok = hmac.compare_digest(body.username, ADMIN_USER) and hmac.compare_digest(body.password, ADMIN_PASSWORD)
    if not ok:
        raise HTTPException(401, "Bad credentials")
    exp = datetime.now(timezone.utc) + timedelta(hours=8)
    return {"token": jwt.encode({"sub": body.username, "exp": exp}, JWT_SECRET, algorithm="HS256")}


@app.post("/customers")
def create_customer(body: CustomerIn, admin: str = Depends(current_admin), db: Session = Depends(get_db)):
    c = Customer(**body.model_dump())
    db.add(c)
    audit(db, admin, "customer.create", body.name)
    db.commit()
    return {"id": c.id}


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int, admin: str = Depends(current_admin), db: Session = Depends(get_db)):
    c = db.get(Customer, customer_id)
    if not c:
        raise HTTPException(404)
    return {
        "customer": {"id": c.id, "name": c.name, "phone": c.phone},
        "account": account_summary(db, c.id),
        "devices": [
            {"id": d.id, "imei": d.imei, "lock_state": d.lock_state, "last_seen": d.last_seen}
            for d in db.scalars(select(Device).where(Device.customer_id == c.id))
        ],
        "installments": [
            {"id": i.id, "seq": i.seq, "due_date": i.due_date, "amount": i.amount, "paid": i.paid}
            for i in db.scalars(select(Installment).where(Installment.customer_id == c.id).order_by(Installment.seq))
        ],
    }


@app.post("/devices")
def enroll_device(body: DeviceIn, admin: str = Depends(current_admin), db: Session = Depends(get_db)):
    if not db.get(Customer, body.customer_id):
        raise HTTPException(404, "Customer not found")
    token = secrets.token_urlsafe(32)  # shown once; only its hash is stored
    d = Device(**body.model_dump(), token_hash=hashlib.sha256(token.encode()).hexdigest())
    db.add(d)
    audit(db, admin, "device.enroll", body.imei)
    db.commit()
    return {"id": d.id, "device_token": token}


@app.post("/customers/{customer_id}/installments")
def create_plan(customer_id: int, body: PlanIn, admin: str = Depends(current_admin), db: Session = Depends(get_db)):
    if not db.get(Customer, customer_id):
        raise HTTPException(404)
    base, extra = divmod(body.total, body.count)
    for n in range(body.count):
        db.add(Installment(
            customer_id=customer_id,
            seq=n + 1,
            due_date=body.first_due + timedelta(days=body.interval_days * n),
            amount=base + (extra if n == body.count - 1 else 0),
        ))
    audit(db, admin, "plan.create", f"customer={customer_id} total={body.total} x{body.count}")
    db.commit()
    return account_summary(db, customer_id)


@app.post("/installments/{installment_id}/payments")
def record_payment(installment_id: int, body: PaymentIn, admin: str = Depends(current_admin), db: Session = Depends(get_db)):
    inst = db.get(Installment, installment_id)
    if not inst:
        raise HTTPException(404)
    if body.amount > inst.amount - inst.paid:
        raise HTTPException(400, "Payment exceeds amount remaining on this installment")
    inst.paid += body.amount
    db.add(Payment(installment_id=inst.id, amount=body.amount, method=body.method, recorded_by=admin))
    audit(db, admin, "payment.record", f"installment={inst.id} amount={body.amount}")
    db.commit()
    return account_summary(db, inst.customer_id)


@app.post("/devices/{device_id}/commands")
def issue_command(device_id: int, body: CommandIn, admin: str = Depends(current_admin), db: Session = Depends(get_db)):
    if not db.get(Device, device_id):
        raise HTTPException(404)
    c = Command(
        device_id=device_id,
        action=body.action,
        reason=body.reason,
        nonce=secrets.token_hex(16),
        issued_by=admin,
        expires_at=now() + timedelta(hours=COMMAND_TTL_HOURS),
    )
    db.add(c)
    audit(db, admin, f"command.{body.action.lower()}", f"device={device_id} reason={body.reason}")
    db.commit()
    return {"command_id": c.id, "expires_at": c.expires_at}


# ----- endpoints the Android app calls -----
@app.post("/device/checkin")
def checkin(device: Device = Depends(current_device), db: Session = Depends(get_db)):
    device.last_seen = now()
    pending = db.scalars(
        select(Command)
        .where(Command.device_id == device.id, Command.acked_at.is_(None), Command.expires_at > now())
        .order_by(Command.id)
    ).all()
    db.commit()
    return {
        "commands": [sign_command(c) for c in pending],
        "account": account_summary(db, device.customer_id),  # drives the visible payment/default screen
        "support_phone": SUPPORT_PHONE,                      # appeal / contact option
    }


@app.post("/device/commands/{command_id}/ack")
def ack(command_id: int, device: Device = Depends(current_device), db: Session = Depends(get_db)):
    c = db.get(Command, command_id)
    if not c or c.device_id != device.id:
        raise HTTPException(404)
    if c.acked_at is None:
        c.acked_at = now()
        device.lock_state = {"LOCK": "LOCKED", "UNLOCK": "UNLOCKED", "RELEASE": "RELEASED"}[c.action]
        audit(db, f"device:{device.id}", "command.ack", f"{c.action} id={c.id}")
        db.commit()
    return {"lock_state": device.lock_state}
