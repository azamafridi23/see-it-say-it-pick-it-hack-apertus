import os
import json
import traceback


from .scene import load_scenes
from .multimodal import llm_processing



def finalizing_output_format(scene, llm_output):
    output = []
    for idx, instruction in enumerate(llm_output, start=1):
        # Get first action pixel
        first_action = instruction["actions_data"][0]
        first_pixel = None if "pixel" not in first_action.keys() else first_action["pixel"]
        instruction_result = {
            "instruction_id": f"{scene}_i{idx}",
            "target_pixel": first_pixel,
            "actions": []
        }
        for action in instruction["actions_data"]:
            instruction_result["actions"].append(action)

        output.append(instruction_result)

    return output


def pipeline(dataset_dir_path):
    # Load scene
    scenes = load_scenes(dataset_dir_path)

    # LLM processing
    final_result = {
        "team": "path_b_wahyu",
        "results": []
    }
    for scene in scenes:
        # if scene not in ["scene_01", "scene_02"]:
        #     continue
        try:
            print(f"Processing {scene}")
            image_scene_path = scenes[scene]["image_scene_path"]
            llm_output = llm_processing(
                scenes[scene]["instruction_data"],
                image_scene_path
            )
            final_result["results"] += finalizing_output_format(scene, llm_output)
            # print(final_result)
            # break
        except:
            traceback.print_exc()
            with open("./result_wahyu.json", "w") as f:
                json.dump(final_result, f, indent=4)


    with open("./result_wahyu.json", "w") as f:
        json.dump(final_result, f, indent=4)



if __name__ == "__main__":
    pipeline("./dataset")