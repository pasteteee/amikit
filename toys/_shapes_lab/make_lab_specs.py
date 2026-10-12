"""Lab variants of new2610 toys with the new shaped pieces: <id>_base.json -> <id>_new.json"""
import json, sys, copy
from pathlib import Path
LAB = Path("D:/Work/amigurumi-handover-2026-10-02/amigurumi-handover/amigurumi-lab/amikit/toys/_shapes_lab")


def load(i):
    return json.loads((LAB / f"{i}_base.json").read_text(encoding="utf-8-sig"))


def part(s, pid):
    return next(p for p in s["parts"] if p["id"] == pid)


def entries(s, pid):
    return [e for e in s["assembly"] if e["part"] == pid]


def save(i, s):
    s["name"] = s["name"] + "_new"
    (LAB / f"{i}_new.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf8")


def dragon():
    s = load("04_pistachio_dragon")
    w = part(s, "wing")
    for k in ("rounds", "end", "stuff", "spoon"):
        w.pop(k, None)
    w["motif"] = {"kind": "half_circle", "rows": 4, "edging": {"kind": "picot", "every": 3}, "sew": "base"}
    e = entries(s, "wing")[0]
    for k in ("front",):
        e.pop(k, None)
    e.update({"attach": "edge", "dir": [1, 0.45, -0.15], "face": [0.2, 0.1, 1], "sink": 1})
    save("04_pistachio_dragon", s)


def dino():
    s = load("10_apricot_dinosaur")
    for pid, rows in (("plate", 5), ("crown_plate", 6), ("small_plate", 4)):
        p = part(s, pid)
        for k in ("rounds", "end", "stuff"):
            p.pop(k, None)
        p["motif"] = {"kind": "half_circle", "rows": rows}
        for e in entries(s, pid):
            d = e.get("dir", [0, 1, 0])
            e.update({"attach": "edge", "sink": 2})
            # plates stand in the plane through the spine: their face looks sideways
            e["face"] = [1, 0, 0] if abs(d[0]) < 0.5 else [0, 0, 1]
    save("10_apricot_dinosaur", s)




def frog():
    s = load("03_matcha_frog")
    c = part(s, "collar")
    c.update({"rounds": ["ring 42", "inc 5 x5"], "end": "open", "stuff": "none"})
    e = entries(s, "collar")[0]
    for k in ("attach", "sink", "conform_parent", "dir"):
        e.pop(k, None)
    e.update({"around": True, "at": {"round": 17}})
    for f in s["face"]:
        if f["on"] == "collar":
            f["at"] = {"round": 3.5, "az": 52}
    save("03_matcha_frog", s)


def puffin():
    s = load("20_cloud_puffin")
    s["parts"] = [p for p in s["parts"] if "collar" not in p["id"] and "scallop" not in p["id"]]
    s["assembly"] = [e for e in s["assembly"] if "collar" not in e["part"] and "scallop" not in e["part"]]
    s["parts"].append({"id": "collar", "name_en": "collar", "name_ru": "воротничок",
                       "rounds": ["ring 42", "inc 4 x3", "shells 7 every 6"], "colour": "powder", "end": "open",
                       "stuff": "none"})
    s["assembly"].append({"part": "collar", "on": "body", "at": {"round": "end"}, "around": True})
    save("20_cloud_puffin", s)


def panda():
    s = load("05_cinnamon_red_panda")
    s["parts"].append({"id": "snood", "name_en": "snood", "name_ru": "снуд", "rounds": ["ring 40", "rib 40 x3"],
                       "colour": "oatmeal", "end": "open", "stuff": "none"})
    s["assembly"].append({"part": "snood", "on": "body", "at": {"round": 18}, "around": True, "dir": [0, 1, 0]})
    save("05_cinnamon_red_panda", s)




def otter():
    s = load("06_shell_otter")
    h = part(s, "head")
    h["colours"] = [z for z in h.get("colours", []) if not ("ellipse" in z and z["ellipse"]["center"][0] < 20)]
    s["parts"].append({"id": "face_mask", "name_en": "face mask", "name_ru": "светлая маска мордочки",
                       "motif": {"kind": "mask", "chain": 15, "rounds": 3, "tall": "tr"}, "colour": "cream"})
    i = next(k for k, e in enumerate(s["assembly"]) if e["part"] == "head") + 1
    s["assembly"].insert(i, {"part": "face_mask", "on": "head", "applique": True, "anchor": "eyes",
                             "offset_mm": [0, -7], "lift": 0.8, "drape": True})
    for e in entries(s, "cheek"):
        e["lift"] = 2.2                      # on top of the mask
    save("06_shell_otter", s)




def capybara():
    s = load("02_pear_capybara")
    b = part(s, "body")
    b["colours"] = [z for z in b.get("colours", []) if z.get("colour") != "sage"]
    s["parts"].append({"id": "kerchief", "name_en": "kerchief", "name_ru": "косынка",
                       "motif": {"kind": "triangle", "width": 26}, "colour": "sage"})
    i = next(k for k, e in enumerate(s["assembly"]) if e["part"] == "head") + 1
    s["assembly"].append({"part": "kerchief", "on": "body", "at": {"round": 19, "az": 0}, "applique": True,
                          "drape": True, "roll": 180, "offset_mm": [0, 3], "lift": 1.0})
    lf = part(s, "pleaf")
    for k in ("rounds", "end", "stuff", "flatten"):
        lf.pop(k, None)
    lf["motif"] = {"kind": "leaf", "chain": 6, "tall": "dc"}
    e = entries(s, "pleaf")[0]
    e.pop("attach", None); e.pop("sink", None)
    e.update({"attach": "edge", "face": [0.3, 0.2, 1], "sink": 1})
    save("02_pear_capybara", s)


def bat():
    s = load("15_berry_bat")
    s["parts"] = [p for p in s["parts"] if p["id"] != "scallop"]
    s["assembly"] = [e for e in s["assembly"] if e["part"] != "scallop"]
    w = part(s, "wing")
    for k in ("rounds", "end", "stuff", "flatten"):
        w.pop(k, None)
    w["motif"] = {"kind": "half_circle", "rows": 5, "edging": {"kind": "shell", "k": 5, "every": 5 if False else 4}}
    e = entries(s, "wing")[0]
    e.pop("front", None)
    e.update({"attach": "edge", "dir": [1, 0.15, -0.35], "face": [0.25, 0.05, 1], "sink": 1})
    s["face"] = [f for f in s["face"] if not (f["type"] == "satin" and f["on"] == "body" and f.get("shape") == "heart")]
    s["parts"].append({"id": "heart", "name_en": "heart", "name_ru": "сердечко", "motif": {"kind": "heart"},
                       "colour": "heart"})
    s["assembly"].append({"part": "heart", "on": "body", "at": {"round": 11, "az": 0}, "applique": True, "lift": 0.8})
    save("15_berry_bat", s)


def bunny():
    s = load("07_clover_bunny")
    s["face"] = [f for f in s["face"] if not (f["type"] == "satin" and f["on"] == "belly")]
    s["parts"].append({"id": "clover_leaf", "name_en": "clover leaf", "name_ru": "листик клевера",
                       "motif": {"kind": "heart"}, "colour": "clover"})
    for k, (rl, dx, dy) in enumerate(((0, 0, 9.5), (120, -8.2, -4.75), (-120, 8.2, -4.75))):
        s["assembly"].append({"part": "clover_leaf", "name": f"clover_{k}", "on": "belly", "at": {"round": "start"},
                              "applique": True, "roll": rl, "offset_mm": [dx, dy], "lift": 0.8})
    save("07_clover_bunny", s)


def quokka():
    s = load("19_honey_quokka")
    hc = part(s, "honeycomb")
    for k in ("rounds", "end", "stuff"):
        hc.pop(k, None)
    hc["motif"] = {"kind": "hexagon", "rounds": 3, "point_up": True}
    hc["colours"] = [{"rounds": [3, 3], "colour": "darkochre"}]
    pos = {"honeycomb_top": (12.5, 0), "honeycomb_l": (9, 13.5), "honeycomb_r": (9, -13.5)}
    for e in entries(s, "honeycomb"):
        r, a = pos[e["name"]]
        e["at"] = {"round": r, "az": a}
    save("19_honey_quokka", s)


def tapir():
    s = load("17_watermelon_tapir")
    bag = part(s, "bag")
    for k in ("rounds", "end", "stuff"):
        bag.pop(k, None)
    bag["motif"] = {"kind": "half_circle", "rows": 5}
    bag["colours"] = [{"rounds": [5, 5], "colour": "sage"}]
    e = entries(s, "bag")[0]
    e["pad"] = 2.0
    e["roll"] = 180
    for f, az in zip([f for f in s["face"] if f["on"] == "bag"], (100, 180, 260)):
        f["at"] = {"round": 2.6, "az": az}
    save("17_watermelon_tapir", s)


def pangolin():
    s = load("18_artichoke_pangolin")
    for pid in ("leaf_dark", "leaf_light"):
        p = part(s, pid)
        for k in ("rounds", "end", "stuff"):
            p.pop(k, None)
        p["motif"] = {"kind": "leaf", "chain": 7, "tall": "dc"}
        for e in entries(s, pid):
            e["roll"] = 180 if not e.get("along") else 180
    save("18_artichoke_pangolin", s)


def panda_cheeks():
    s = load("05_cinnamon_red_panda")
    s["parts"].append({"id": "snood", "name_en": "snood", "name_ru": "снуд", "rounds": ["ring 48", "rib 48 x4"],
                       "colour": "oatmeal", "end": "open", "stuff": "none"})
    s["assembly"].append({"part": "snood", "on": "body", "at": {"round": 16.5}, "around": True, "dir": [0, 1, 0]})
    for e in entries(s, "cheek_patch"):
        e.pop("at", None)
        e.update({"anchor": "eye", "offset_mm": [-20, -15]})
    save("05_cinnamon_red_panda", s)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
