def evaluate_growth(before, after):
    growth = ((after - before) / before) * 100
    return {"Growth Percentage": growth}
