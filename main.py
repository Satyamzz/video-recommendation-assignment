from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
import random



from fetch import videos
from fetch import user_data
# Convert to dict for fast lookup
user_data_dict = {u["username"]: u for u in user_data}

# Mood → category mapping
mood_to_category = {
    "happy": ["sports", "gym"],
    "motivated": ["finance", "gym"],
    "relaxed": ["finance"],
    "default": ["finance", "gym", "sports"],  # <- new default mood
}


# ------------------------
# Feed API
# ------------------------
app = FastAPI()
@app.get("/feed")
def get_feed(username: str, mood: Optional[str] = None) -> List[dict]:
    user = user_data_dict.get(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history = user.get("viewed_videos", [])

    # Cold-start (no history)
    if not history:
        if mood:
            categories = mood_to_category.get(mood.lower(), [])
            recs = [v for v in videos if v["category"] in categories]
            recs = sorted(recs, key=lambda v: v["likes"], reverse=True)
            if not recs:  # fallback
                return sorted(videos, key=lambda v: v["views"], reverse=True)[:3]
            return recs[:3]
        else:
            # cold start, no mood → popular
            return sorted(videos, key=lambda v: v["views"], reverse=True)[:3]

    # Existing user logic (recommend based on watched categories)
    watched_categories = {v["category"] for v in videos if v["id"] in history}
    recs = [v for v in videos if v["category"] in watched_categories and v["id"] not in history]

    if not recs:
        recs = sorted(videos, key=lambda v: v["views"], reverse=True)

    return random.sample(recs, min(3, len(recs)))


@app.get("/videos-by-category")
def get_videos_by_category(category: str = Query(..., description="Category to filter by")) -> List[dict]:
    filtered_videos = [v for v in videos if v["category"].lower() == category.lower()]

    if not filtered_videos:
        raise HTTPException(status_code=404, detail=f"No videos found for category '{category}'")
    return filtered_videos