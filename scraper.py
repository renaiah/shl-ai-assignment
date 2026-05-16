import json

def load_catalog():
    with open("catalog.json") as f:
        raw = json.load(f)
    
    assessments = []
    for item in raw:
        # Map test type from "keys" field
        keys = item.get("keys", [])
        test_type = map_test_type(keys)
        
        name = item.get("name", "").strip()
        url = item.get("link", "").strip()
        description = item.get("description", name)
        job_levels = item.get("job_levels", [])
        languages = item.get("languages", [])
        duration = item.get("duration", "")
        remote = item.get("remote", "")
        adaptive = item.get("adaptive", "")
        
        if name and url:
            assessments.append({
                "name": name,
                "url": url,
                "description": description,
                "test_type": test_type,
                "job_levels": job_levels,
                "languages": languages,
                "duration": duration,
                "remote": remote,
                "adaptive": adaptive,
                "keys": keys
            })
    
    # Save cleaned version
    with open("catalog_clean.json", "w") as f:
        json.dump(assessments, f, indent=2)
    
    print(f"Loaded {len(assessments)} assessments")
    return assessments

def map_test_type(keys):
    if not keys:
        return "K"
    keys_lower = [k.lower() for k in keys]
    if any("personality" in k or "behavior" in k for k in keys_lower):
        return "P"
    elif any("ability" in k or "aptitude" in k for k in keys_lower):
        return "A"
    elif any("situational" in k or "biodata" in k for k in keys_lower):
        return "S"
    elif any("competenc" in k or "360" in k or "development" in k for k in keys_lower):
        return "C"
    return "K"

if __name__ == "__main__":
    load_catalog()