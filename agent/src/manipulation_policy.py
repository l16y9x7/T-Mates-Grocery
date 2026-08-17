"""Cross-task policies for exceptional shelf manipulation geometry."""

SPECIAL_SHELF_NUDGE_PRODUCT = "外星人电解质水白桃口味0糖"


def initial_shelf_nudge_direction(product_name: str, hand: str) -> str | None:
    if product_name != SPECIAL_SHELF_NUDGE_PRODUCT:
        return None
    if hand.upper() == "LEFT":
        return "right"
    if hand.upper() == "RIGHT":
        return "left"
    return None
