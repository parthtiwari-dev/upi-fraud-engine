# Engineering Learnings

## Concept: Label Delay & The "Omniscient" Trap
**Why it matters:**
In Kaggle/Tutorials, you have the target variable `y` immediately. In banking, `y` arrives days later.

**The Mistake:**
If I train my model using today's labels to predict today's fraud, I am "cheating." In production, those labels won't exist yet.

**The Fix:**
My training set must be strictly older than my inference set by at least 48 hours.
