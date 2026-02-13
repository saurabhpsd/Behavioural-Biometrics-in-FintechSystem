// =====================================================
// FULL SERVER.JS — Behavioural Biometrics + Transaction Fraud
// =====================================================

const express = require("express");
const path = require("path");
const fetch = require("node-fetch");
const sqlite3 = require("sqlite3").verbose();
const bcrypt = require("bcrypt");
const fs = require("fs");

const app = express();
const PORT = 3000;

// =====================================================
// DATABASE
// =====================================================
const DB_DIR = path.join(__dirname, "db");
const DB_FILE = path.join(DB_DIR, "users.db");

if (!fs.existsSync(DB_DIR)) fs.mkdirSync(DB_DIR);

const db = new sqlite3.Database(DB_FILE, (err) => {
  if (err) return console.error("DB open error:", err);
  console.log("SQLite DB ready:", DB_FILE);
});

db.exec(
  `
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 username TEXT UNIQUE,
 password_hash TEXT,
 role TEXT,
 last_login TEXT
);`,
  (err) =>
    err ? console.error(err) : console.log("DB Schema Verified")
);

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// =====================================================
// EVENT STORES
// =====================================================
const loginEvents = [];
const transactionEvents = [];
const lastDecisionCache = {}; // rapid retry prevention

function logLoginEvent(obj) {
  loginEvents.unshift({
    timestamp: new Date().toLocaleString(),
    ...obj,
  });
  if (loginEvents.length > 50) loginEvents.pop();
}

function logTransactionEvent(obj) {
  transactionEvents.unshift({
    timestamp: new Date().toLocaleString(),
    ...obj,
  });
  if (transactionEvents.length > 50) transactionEvents.pop();
}

// =====================================================
// CALL PYTHON API (GENERIC HELPER)
// =====================================================
async function callPythonAPI(endpoint, payload) {
  try {
    const res = await fetch(`http://127.0.0.1:5001${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return await res.json();
  } catch (e) {
    console.error("Python API offline:", e);
    return {
      allowed: false,
      riskScore: 0.9,
      decision: "BLOCK",
      model: "offline",
      explanation: "Python Engine Offline",
    };
  }
}

// =====================================================
// REGISTRATION
// =====================================================
app.post("/api/register", async (req, res) => {
  const { username, password, role } = req.body;
  if (!username || !password || !role)
    return res.status(400).json({ ok: false });

  const hash = await bcrypt.hash(password, 10);
  db.run(
    "INSERT INTO users(username, password_hash, role) VALUES (?,?,?)",
    [username, hash, role],
    (err) => {
      if (err) return res.status(409).json({ ok: false });
      res.json({ ok: true });
    }
  );
});

// =====================================================
// CUSTOMER LOGIN (Fraud-Protected)
// =====================================================
app.post("/api/customer-login", async (req, res) => {
  const { username, password, behaviour } = req.body;

  db.get(
    "SELECT * FROM users WHERE username=?",
    [username],
    async (err, row) => {
      if (!row || row.role !== "customer")
        return res.json({
          allowed: false,
          message: "Invalid credentials",
        });

      const ok = await bcrypt.compare(password, row.password_hash);
      if (!ok)
        return res.json({
          allowed: false,
          message: "Invalid credentials",
        });

      // --- Mandatory Behaviour ---
      if (!behaviour || Object.keys(behaviour).length === 0) {
        logLoginEvent({
          username,
          role: "Customer",
          riskScore: 1.0,
          decision: "block",
          explanation: "Behaviour missing",
          behaviour,
          model: "missing_data",
        });
        return res.json({
          allowed: false,
          message: "Behaviour missing. Login blocked.",
        });
      }

      // --- Hardening ---
      if (behaviour.illegal_edit_detected) {
        logLoginEvent({
          username,
          role: "Customer",
          riskScore: 1.0,
          decision: "block",
          explanation: "Backspace/Edit Tampering",
          behaviour,
          model: "policy",
        });
        return res.json({
          allowed: false,
          message: "Login blocked by policy",
        });
      }

      // --- Retry Cooldown ---
      if (lastDecisionCache[username]) {
        const last = lastDecisionCache[username];
        if (
          last.decision === "block" &&
          Date.now() - last.time < 5000
        ) {
          logLoginEvent({
            username,
            role: "Customer",
            riskScore: 1.0,
            decision: "block",
            explanation: "Retry cooldown",
            behaviour,
            model: "cooldown",
          });
          return res.json({
            allowed: false,
            message: "Retry cooldown active",
          });
        }
      }

      // --- Python Fraud Engine ---
      const fraud = await callPythonAPI("/api/hmog-login", {
        username,
        role: "customer",
        behaviour,
      });

      const allowed = fraud.riskScore < 0.7;

      logLoginEvent({
        username,
        role: "Customer",
        riskScore: fraud.riskScore,
        decision: allowed ? "allow" : "block",
        explanation: fraud.reasons
          ? fraud.reasons.join(", ")
          : "No explanation",
        behaviour,
        model: fraud.model,
      });

      lastDecisionCache[username] = {
        decision: allowed ? "allow" : "block",
        time: Date.now(),
      };

      if (!allowed)
        return res.json({
          allowed: false,
          message: "Login blocked by Identity Protection System",
        });

      db.run("UPDATE users SET last_login=datetime('now') WHERE id=?", [
        row.id,
      ]);

      res.json({ allowed: true, redirect: "/user_dashboard.html" });
    }
  );
});

// =====================================================
// EMPLOYEE LOGIN (No Fraud Check Required)
// =====================================================
app.post("/api/employee-login", (req, res) => {
  const { username, password } = req.body;

  db.get(
    "SELECT * FROM users WHERE username=?",
    [username],
    async (err, row) => {
      if (!row) return res.json({ allowed: false });
      const ok = await bcrypt.compare(password, row.password_hash);
      if (!ok) return res.json({ allowed: false });

      res.json({ allowed: true, redirect: "/employee_dashboard.html" });
    }
  );
});

// =====================================================
// ENROLLMENT FOR LOGIN BIOMETRICS
// =====================================================
app.post("/api/enroll", async (req, res) => {
  try {
    const py = await callPythonAPI("/api/save-training-data", req.body);
    res.json(py);
  } catch {
    res.status(500).json({ ok: false });
  }
});

// =====================================================
// TRANSACTION BEHAVIOUR TRAINING
// =====================================================
app.post("/api/save-transaction-data", async (req, res) => {
  try {
    const py = await callPythonAPI(
      "/api/save-transaction-data",
      req.body
    );
    res.json(py);
  } catch {
    res.status(500).json({ ok: false });
  }
});
app.post("/api/save-transaction-data", async (req, res) => {
  const result = await callPythonAPI("/api/save-transaction-data", req.body);
  res.json(result);
});

app.post("/api/transaction-check", async (req, res) => {
  const result = await callPythonAPI("/api/transaction-check", req.body);

  transactionEvents.unshift({
    timestamp: new Date().toLocaleString(),
    ...req.body,
    riskScore: result.riskScore,
    decision: result.decision,
    model: "transaction"
  });

  res.json(result);
});

app.get("/api/transaction-events", (req, res) => {
  res.json(transactionEvents);
});

// =====================================================
// TRANSACTION FRAUD CHECK
// =====================================================
app.post("/api/transaction-check", async (req, res) => {
  const { owner, acct, amt, behaviour } = req.body;

  const result = await callPythonAPI("/api/transaction-check", {
    owner,
    acct,
    amt,
    behaviour,
  });

  logTransactionEvent({
    owner,
    acct,
    amt,
    riskScore: result.riskScore,
    decision: result.allowed ? "allow" : "block",
    model: result.model,
  });

  res.json(result);
});

// =====================================================
// API ROUTES — DASHBOARD DATA FETCH
// =====================================================
app.get("/api/login-events", (req, res) => {
  res.json(loginEvents);
});

app.get("/api/transaction-events", (req, res) => {
  res.json(transactionEvents);
});

// =====================================================
// ROUTING FOR ALL PAGES
// =====================================================
app.get("/", (req, res) =>
  res.sendFile(path.join(__dirname, "public", "login.html"))
);

app.get("/enroll", (req, res) =>
  res.sendFile(path.join(__dirname, "public", "enroll.html"))
);

app.get("/transaction", (req, res) =>
  res.sendFile(path.join(__dirname, "public", "transaction.html"))
);

app.get("/transaction_success.html", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "transaction_success.html"));
});


app.get("/transaction_blocked", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "transaction_blocked.html"))
});


app.get("/train_transaction", (req, res) =>
  res.sendFile(path.join(__dirname, "public", "train_transaction.html"))
);
app.get("/employee_transaction.html", (req, res) =>
  res.sendFile(path.join(__dirname, "public", "employee_transaction.html"))
);

app.get("/transaction_events", (req, res) =>
  res.sendFile(
    path.join(__dirname, "public", "transaction_events.html")
  )
);

// =====================================================
// START SERVER
// =====================================================
app.listen(PORT, () =>
  console.log(`Node.js server running at http://localhost:${PORT}`)
);
