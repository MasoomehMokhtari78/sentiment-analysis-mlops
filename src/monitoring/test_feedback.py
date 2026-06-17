from feedback import log_prediction

print("Testing feedback collection...")

log_prediction("this movie was amazing", "positive", "positive", 0.92)
log_prediction("very bad experience", "negative", "positive", 0.88)

print("Feedback saved with labels.")