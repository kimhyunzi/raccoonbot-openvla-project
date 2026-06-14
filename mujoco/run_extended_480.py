from raccoon_extended_tasks_dataset import collect_extended_task_dataset

collect_extended_task_dataset(
    xml_path="Raccoon_colored_cylinder.xml",
    dataset_root="raccoon_extended_tasks_480",
    tasks=("lift", "push", "pick_place"),
    colors=("red", "blue", "green", "yellow"),
    episodes_per_task_color=40,
    keep_failed=False,
    use_viewer=False,
    camera_name="front_view",
    speed=150,
    settle_seconds_per_action=1.0,
    initial_settle_seconds=0.1,
    hz=10,
    touch_threshold=0.1,
    seed=20260613,
    max_attempts=3000,
    object_x_range=(-0.10, 0.10),
    object_y_range=(0.16, 0.25),
    min_object_distance=0.035,
)
