CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name
    in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    index: class_name
    for class_name, index
    in CLASS_TO_IDX.items()
}