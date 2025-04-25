import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size
    print(f"Image size: {img_width}x{img_height}")
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    print(f"Scale factors: {scale_x}, {scale_y}")
    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        print(f"Original coordinates: {x1}, {y1}, {x2}, {y2}")
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        print(f"Scaled coordinates: {x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}")

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    with open(info_path) as f:
        info = json.load(f)

    frame_detections = info["detections"][view_index]
    # ego_kart_id = info.get("ego_kart_id", 0)  # Default to 0 if not present
    # track_name = info.get("track", "unknown")

    karts_objects = []
    closest_kart_distance = float('inf')
    closest_kart_index = -1

    image_center_x = img_width / 2
    image_center_y = img_height / 2

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    min_distance = 99999999
    ego_kart_id = None
    for i, detection in enumerate(frame_detections):
        object_class_id, kart_id, x1, y1, x2, y2 = map(int, detection)

        # Skip if the detection object is not a kart 
        if object_class_id != 1: 
            continue

        # Scale coordinates
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Calculate center of the kart
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        # Skip if bounding box is too small
        # if (x2_scaled - x1_scaled) < 5 or (y2_scaled - y1_scaled) < 5:
        #     continue

        distance_from_img_center = np.sqrt((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2)

        if(distance_from_img_center < min_distance):
            min_distance = distance_from_img_center
            ego_kart_id = kart_id
        # Check if kart is within image boundaries
        # if not (0 <= x1_scaled <= img_width and 0 <= x2_scaled <= img_width and 0 <= y1_scaled <= img_height and 0 <= y2_scaled <= img_height):
        #     continue

        # Determine kart name
        # kart_name = "ego car" if track_id == ego_kart_id else OBJECT_TYPES[class_id].lower()

        karts_objects.append({
            "instance_id": kart_id,
            "kart_name": info["karts"][kart_id] if kart_id < len(info["karts"]) else "unknown",
            "center": (center_x, center_y),
            "is_ego_kart": kart_id == ego_kart_id,
            "distance_to_center": distance_from_img_center  # Store distance for comparison
        })

        # if dist_to_center < closest_kart_distance:
        #     closest_kart_distance = dist_to_center
        #     closest_kart_index = len(karts) - 1

    # Mark the closest kart (if any)
    #if closest_kart_index != -1:
    #    karts[closest_kart_index]["is_center_kart"] = True
    #else:
    #    for kart in karts:
    #      kart["is_center_kart"] = False
    print(f"karts_objects: {karts_objects}")
    return karts_objects

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa_pairs = []
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    ego_kart_name = next((kart["kart_name"] for kart in karts if kart["is_ego_kart"]), "the ego car")

    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_kart_name
    })

    # 2. Total karts question
    num_karts = len(karts)
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(num_karts)
    })

    # 3. Track information questions
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    if num_karts > 1:
        for kart in karts:
            if kart["is_ego_kart"]:
                continue  # Skip ego kart

            # 4. Relative position questions for each kart
            kart_name = kart["kart_name"]
            ego_x, ego_y = next((k["center"] for k in karts if k["is_ego_kart"]), (img_width / 2, img_height / 2))  # Default to center if ego not found
            kart_x, kart_y = kart["center"]

            relative_position = []
            if kart_x <= ego_x:
                relative_position.append("left")
            elif kart_x > ego_x:
                relative_position.append("right")

            if kart_y <= ego_y:
                relative_position.append("front")
            elif kart_y > ego_y:
                relative_position.append("behind")

            position_answer = " and ".join(relative_position) if relative_position else "same location"

            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "left" if kart_x < ego_x else "right" if kart_x > ego_x else "left"
            })
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "behind" if kart_y > ego_y else "front" if kart_y < ego_y else "front"
            })
            qa_pairs.append({
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": position_answer
            })

        # 5. Counting questions
        left_count = sum(1 for kart in karts if not kart["is_ego_kart"] and kart["center"][0] <= ego_x)
        right_count = sum(1 for kart in karts if not kart["is_ego_kart"] and kart["center"][0] > ego_x)
        behind_count = sum(1 for kart in karts if not kart["is_ego_kart"] and kart["center"][1] > ego_y)
        front_count = sum(1 for kart in karts if not kart["is_ego_kart"] and kart["center"][1] <= ego_y)

        qa_pairs.append({
            "question": "How many karts are to the left of the ego car?",
            "answer": str(left_count)
        })
        qa_pairs.append({
            "question": "How many karts are to the right of the ego car?",
            "answer": str(right_count)
        })
        qa_pairs.append({
            "question": "How many karts are in front of the ego car?",
            "answer": str(front_count)
        })
        qa_pairs.append({
            "question": "How many karts are behind the ego car?",
            "answer": str(behind_count)
        })

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""
def build_full_dataset(split: str = "train", view_indices: list[int] = [0, 1, 2]):
    """
    Generate QA dataset for all info files in a split directory and save to a single .json file.
    
    Args:
        split: Dataset split (e.g., 'train')
        view_indices: List of view indices to process per frame
    """
    import os
    dataset = []

    info_dir = Path(f"../data/{split}")
    out_dir = info_dir
    out_file = out_dir / "expanded_qa_pairs.json"

    info_files = sorted(info_dir.glob("*_info.json"))
    print(f"Found {len(info_files)} info files.")

    for info_file in info_files:
        for view_index in view_indices:
            try:
                qa = generate_qa_pairs(str(info_file), view_index)
                base_name = info_file.stem.replace("_info", "")
                image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
                for pair in qa:
                    pair["image_file"] = image_file
                dataset.extend(qa)
            except Exception as e:
                print(f"Error in {info_file} view {view_index}: {e}")
                continue

    print(f"Generated {len(dataset)} QA pairs.")

    # Save
    with open(out_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved expanded dataset to {out_file}")


def main():
    fire.Fire({"check": check_qa_pairs, "draw": draw_detections, "extract": extract_kart_objects, "generate": generate_qa_pairs, "build": build_full_dataset})


if __name__ == "__main__":
    main()
