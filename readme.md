## 🧑‍💻 Team Git Workflow (Simple)

Follow these steps to work on your own branch safely.

---

### 1️⃣ Go to the repository

```bash
git clone https://github.com/EunbiYoon/GTBench.git
```

---

### 2️⃣ Access to your branch

```bash
git fetch origin        # Get all remote branches
git branch -a           # Check all branches
git checkout eunbi     # Move to YOUR branch
```

---

### 3️⃣ Pull latest updates

```bash
git pull origin eunbi
```

---

### 4️⃣ Do your work

Edit / add files.

---

### 5️⃣ Stage changes

```bash
git add .
```

---

### 6️⃣ Commit

```bash
git commit -m "Your message"
```

---

### 7️⃣ Push to your branch

```bash
git push origin eunbi
```

---

## ⚠️ Rules

* Only push to **your branch**
* Do NOT push to `main` / `master`
* Do NOT modify other branches
* Always `pull` before `push`

---
