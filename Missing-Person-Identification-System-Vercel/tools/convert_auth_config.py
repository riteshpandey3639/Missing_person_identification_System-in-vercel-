import json, sys, yaml
from pathlib import Path
p=Path(sys.argv[1] if len(sys.argv)>1 else "login_config.yml")
cfg=yaml.safe_load(p.read_text(encoding="utf-8"))
users={}
for username, u in cfg["credentials"]["usernames"].items():
    users[username]={
        "name":u.get("name",""),
        "email":u.get("email",""),
        "city":u.get("city",""),
        "area":u.get("area",""),
        "role":u.get("role","Officer"),
        "password_hash":u["password"],
    }
print(json.dumps(users,separators=(",",":")))
