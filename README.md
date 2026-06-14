# RaccoonBot OpenVLA Multi-task Manipulation

본 프로젝트는 RaccoonBot 로봇팔에 OpenVLA 기반 Vision-Language-Action 모델을 적용하여, 자연어 명령에 따라 다양한 물체 조작 동작을 수행하도록 확장한 프로젝트이다.

기존 grasp 중심 파이프라인을 기반으로, MuJoCo 시뮬레이션 환경에서 multi-task demonstration dataset을 생성하고, 이를 OpenVLA-7B 모델에 LoRA fine-tuning하였다. 이후 fine-tuned 모델이 예측한 7D action을 RaccoonBot의 4DOF arm 구조에 맞게 변환하여 MuJoCo inference 및 실제 로봇 실행 구조와 연결하였다.

## Demo Videos

아래 영상은 fine-tuned OpenVLA 모델을 이용하여 RaccoonBot이 자연어 instruction에 따라 4가지 조작 동작을 수행하는 결과이다.

### 1. Grasp

Instruction example:

```text
grasp the red cylinder
```

https://github.com/user-attachments/assets/GRASP_VIDEO_ID

또는 repository 내부 영상 파일을 사용할 경우:

```markdown
<video src="assets/videos/grasp.mp4" controls width="600"></video>
```

### 2. Lift

Instruction example:

```text
lift the blue cylinder
```

https://github.com/user-attachments/assets/LIFT_VIDEO_ID

또는 repository 내부 영상 파일을 사용할 경우:

```markdown
<video src="assets/videos/lift.mp4" controls width="600"></video>
```

### 3. Push

Instruction example:

```text
push the green cylinder forward
```

https://github.com/user-attachments/assets/PUSH_VIDEO_ID

또는 repository 내부 영상 파일을 사용할 경우:

```markdown
<video src="assets/videos/push.mp4" controls width="600"></video>
```

### 4. Pick and Place

Instruction example:

```text
pick and place the yellow cylinder
```

https://github.com/user-attachments/assets/PICK_PLACE_VIDEO_ID

또는 repository 내부 영상 파일을 사용할 경우:

```markdown
<video src="assets/videos/pick_place.mp4" controls width="600"></video>
```

## Project Overview

OpenVLA는 camera image와 natural language instruction을 입력으로 받아 로봇 action을 예측하는 Vision-Language-Action 모델이다. 그러나 특정 로봇 플랫폼에 적용하기 위해서는 해당 로봇의 task, observation format, action space에 맞는 dataset과 inference pipeline이 필요하다.

본 프로젝트에서는 RaccoonBot이 다음과 같은 multi-task manipulation을 수행할 수 있도록 OpenVLA pipeline을 확장하였다.

| Task           | Description                    | Example instruction                  |
| -------------- | ------------------------------ | ------------------------------------ |
| Grasp          | target object를 잡는 동작           | `grasp the red cylinder`             |
| Lift           | target object를 잡고 들어올리는 동작     | `lift the blue cylinder`             |
| Push           | target object를 앞으로 미는 동작       | `push the green cylinder forward`    |
| Pick and Place | target object를 들어 옆 위치로 옮기는 동작 | `pick and place the yellow cylinder` |

## Overall Pipeline

본 프로젝트의 전체 pipeline은 다음과 같이 구성된다.

```text
MuJoCo Demonstration Collection
        ↓
Multi-task Dataset Generation
        ↓
TFDS / RLDS Dataset Conversion
        ↓
OpenVLA-7B LoRA Fine-tuning
        ↓
MuJoCo Inference
        ↓
7D Action to RaccoonBot 4DOF Mapping
        ↓
Execution Logging and Result Analysis
```

## Key Features

* MuJoCo 기반 RaccoonBot manipulation 환경 구성
* Grasp, lift, push, pick-and-place multi-task dataset 생성
* 색상 기반 target object instruction 구성
* OpenVLA-7B LoRA fine-tuning
* OpenVLA 7D action을 RaccoonBot 4DOF arm 구조에 맞게 변환
* MuJoCo inference 결과 저장
* Step별 action trace, timing trace, distance trace logging
* 실제 RaccoonBot 실행을 위한 real robot client 구조 구현

## Dataset

기존 grasp dataset을 baseline으로 유지하고, 추가적으로 lift, push, pick_place task를 생성하였다. 각 episode에는 red, blue, green, yellow cylinder가 동시에 배치되며, 자연어 instruction에 포함된 색상이 target object를 결정한다.

최종 dataset 구성은 다음과 같다.

| Task           | Episodes |
| -------------- | -------: |
| Grasp          |      400 |
| Lift           |      160 |
| Push           |      160 |
| Pick and Place |      160 |
| Total          |      880 |

각 episode는 다음과 같은 정보를 포함한다.

```text
episode_id
instruction
task_type
target_color
target_body_name
goal_xy
object_pose
ee_pose
joint_angles
gripper_state
action
image observation
success flag
```

## Instruction Templates

각 task는 여러 자연어 instruction template으로 구성된다. 이를 통해 모델이 동일한 동작을 다양한 문장 표현으로 학습할 수 있도록 하였다.

### Lift

```text
lift the {color} cylinder
pick up the {color} cylinder
raise the {color} cylinder
grasp and lift the {color} cylinder
pick the {color} cylinder up from the table
```

### Push

```text
push the {color} cylinder forward
move the {color} cylinder forward
slide the {color} cylinder forward
nudge the {color} cylinder away from the robot
push the {color} object away from the robot
```

### Pick and Place

```text
pick and place the {color} cylinder
move the {color} cylinder to the side
pick up the {color} cylinder and place it nearby
relocate the {color} cylinder to the side
grasp the {color} cylinder and put it down on the side
```

## TFDS / RLDS Dataset Structure

OpenVLA fine-tuning을 위해 raw MuJoCo dataset을 TFDS/RLDS 형식으로 변환하였다. 각 step은 image observation, robot state, action, reward, terminal flag, language instruction을 포함한다.

```text
observation.image  : RGB image, shape = (256, 256, 3)
observation.state  : robot state, shape = (8,)
action             : OpenVLA action, shape = (7,)
language_instruction
reward
discount
is_first
is_last
is_terminal
```

OpenVLA action은 다음과 같은 7D format을 사용한다.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
```

RaccoonBot은 4DOF arm 구조를 사용하므로, rotation delta는 0으로 채우고 position delta와 gripper command를 중심으로 학습 및 실행하였다.

```text
action = [dx, dy, dz, 0, 0, 0, gripper_cmd]
```

## Fine-tuning Setup

OpenVLA-7B 모델을 base model로 사용하고, RaccoonBot multi-task dataset에 대해 LoRA fine-tuning을 수행하였다.

| Item                  | Value                |
| --------------------- | -------------------- |
| Base model            | `openvla/openvla-7b` |
| Dataset               | `raccoon_pick_place` |
| Total episodes        | 880                  |
| Train / Val split     | 792 / 88             |
| Fine-tuning method    | LoRA                 |
| LoRA rank             | 32                   |
| Batch size            | 8                    |
| Gradient accumulation | 2                    |
| Effective batch size  | 16                   |
| Learning rate         | 5e-4                 |
| Max steps             | 18,000               |
| Save interval         | 3,000 steps          |

학습은 max step 18,000까지 수행되었으며, 최종 fine-tuned model은 `openvla-runs` 하위의 model directory에 저장된다.

## Inference

Inference 단계에서는 MuJoCo camera image와 natural language instruction을 OpenVLA server로 전달하고, 모델이 예측한 7D action을 RaccoonBot 실행 구조에 맞게 변환한다.

```text
Image observation + Language instruction
        ↓
OpenVLA server
        ↓
7D action prediction
        ↓
Delta position clipping
        ↓
Target end-effector position
        ↓
RaccoonBot 4DOF inverse kinematics
        ↓
MuJoCo execution
```

### Action Mapping

OpenVLA가 출력한 action은 다음과 같다.

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
```

RaccoonBot에서는 이를 현재 end-effector position에 대한 delta position으로 해석한다.

```text
target_x = current_x + clipped_dx
target_y = current_y + clipped_dy
target_z = current_z + clipped_dz
```

이후 target position은 RaccoonBot의 safe workspace 안으로 제한되며, inverse kinematics를 통해 4DOF joint command로 변환된다.

```text
OpenVLA 7D action
        ↓
clipped delta xyz
        ↓
target xyz
        ↓
4DOF IK
        ↓
joint command + gripper command
```

Gripper command는 threshold 0.5를 기준으로 open/close를 결정한다.

```text
gripper_cmd >= 0.5 → close
gripper_cmd < 0.5  → open
```

## MuJoCo Inference Usage

먼저 OpenVLA server를 실행한다.

```bash
MODEL_DIR="/data/2023741061/code/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--raccoon-880ep-multitask-18h-s3000--image_aug"

python vla-scripts/deploy.py \
  --pretrained_checkpoint "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000
```

이후 MuJoCo client를 실행한다.

```bash
cd /data/2023741061/code/Raccoonbot_Openvla/Mujoco

python openvla_multicolor_client_enhanced.py \
  --server_url http://127.0.0.1:8000 \
  --task lift \
  --target_color red \
  --instruction "lift the red cylinder" \
  --execution_preset safe \
  --max_steps 30 \
  --episode_id 1
```

Task별 실행 예시는 다음과 같다.

### Grasp

```bash
python openvla_multicolor_client_enhanced.py \
  --server_url http://127.0.0.1:8000 \
  --task grasp \
  --target_color red \
  --instruction "grasp the red cylinder" \
  --execution_preset safe \
  --max_steps 30 \
  --episode_id 1
```

### Lift

```bash
python openvla_multicolor_client_enhanced.py \
  --server_url http://127.0.0.1:8000 \
  --task lift \
  --target_color blue \
  --instruction "lift the blue cylinder" \
  --execution_preset safe \
  --max_steps 30 \
  --episode_id 2
```

### Push

```bash
python openvla_multicolor_client_enhanced.py \
  --server_url http://127.0.0.1:8000 \
  --task push \
  --target_color green \
  --instruction "push the green cylinder forward" \
  --execution_preset safe \
  --max_steps 30 \
  --episode_id 3
```

### Pick and Place

```bash
python openvla_multicolor_client_enhanced.py \
  --server_url http://127.0.0.1:8000 \
  --task pick_place \
  --target_color yellow \
  --instruction "pick and place the yellow cylinder" \
  --execution_preset safe \
  --max_steps 30 \
  --episode_id 4
```

## Execution Presets

Inference client는 실행 속도와 안정성에 따라 preset을 선택할 수 있다.

| Preset     | Speed | Settle time | Max delta xyz | Description |
| ---------- | ----: | ----------: | ------------: | ----------- |
| `safe`     |    70 |      0.80 s |       0.005 m | 안정성 중심      |
| `balanced` |    80 |      0.45 s |       0.008 m | 안정성과 속도의 균형 |
| `fast`     |    95 |      0.20 s |       0.012 m | 빠른 실행 중심    |

초기 테스트에서는 `safe` preset을 사용하는 것을 권장한다.

## Output Logs

MuJoCo inference 결과는 episode 단위로 저장된다.

```text
rollout_outputs_enhanced/
  {task}_{color}_{preset}_episode_000001/
    frame_000000.png
    frame_000001.png
    ...
    action_trace.jsonl
    action_trace.csv
    summary.json
    summary.md
    timing_trace.png
    distance_trace.png
    rollout_meta.json
```

각 파일의 역할은 다음과 같다.

| File                 | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| `frame_*.png`        | step별 MuJoCo camera frame                                   |
| `action_trace.csv`   | step별 action, target position, gripper command, retry count |
| `action_trace.jsonl` | raw action trace log                                        |
| `summary.json`       | episode-level quantitative summary                          |
| `summary.md`         | readable summary report                                     |
| `timing_trace.png`   | inference/execution time plot                               |
| `distance_trace.png` | end-effector와 target object 사이 거리 변화 plot                   |
| `rollout_meta.json`  | rollout configuration metadata                              |

## Real Robot Client

본 프로젝트에서는 MuJoCo inference뿐 아니라 실제 RaccoonBot 실행을 위한 client 구조도 구현하였다.

Real robot client는 OpenVLA가 예측한 7D action을 RaccoonBot command로 변환하여 실제 로봇 제어 구조와 연결한다.

기본 실행 흐름은 다음과 같다.

```text
OpenVLA 7D action
        ↓
env.execute_delta_action7()
        ↓
target xyz + gripper command
        ↓
RealRaccoonController IK
        ↓
robot joint command
```

다만 본 프로젝트의 주요 실험은 MuJoCo 환경에서 수행되었으며, physical robot client는 OpenVLA action을 실제 RaccoonBot 실행 구조로 연결하기 위한 확장 기능으로 구현하였다.

## Results

본 프로젝트에서는 fine-tuned OpenVLA 모델을 사용하여 4가지 manipulation task에 대한 MuJoCo inference를 수행하였다.

| Task           | Target object   | Instruction                          | Result  |
| -------------- | --------------- | ------------------------------------ | ------- |
| Grasp          | Red cylinder    | `grasp the red cylinder`             | Success |
| Lift           | Blue cylinder   | `lift the blue cylinder`             | Success |
| Push           | Green cylinder  | `push the green cylinder forward`    | Success |
| Pick and Place | Yellow cylinder | `pick and place the yellow cylinder` | Success |

Inference 결과는 before/after frame, action trace, timing trace, distance trace를 통해 분석하였다.

## Project Structure

```text
Raccoonbot_Openvla/
  Mujoco/
    openvla_multicolor_client_enhanced.py
    openvla_multicolor_client_real_robot_enhanced.py
    raccoon_extended_tasks_dataset.py
    rollout_outputs_enhanced/

  openvla/
    openvla-runs/
      openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--raccoon-880ep-multitask-18h-s3000--image_aug/

  assets/
    videos/
      grasp.mp4
      lift.mp4
      push.mp4
      pick_place.mp4
```

## Requirements

본 프로젝트는 다음 환경에서 수행되었다.

```text
Python
PyTorch
Transformers
OpenVLA
MuJoCo
TensorFlow Datasets
RLDS
NumPy
OpenCV
Matplotlib
```

CUDA GPU 환경에서 OpenVLA inference 및 fine-tuning을 수행하는 것을 권장한다.

## Conclusion

본 프로젝트에서는 RaccoonBot에 OpenVLA 기반 VLA pipeline을 적용하고, 기존 grasp 중심 구조를 grasp, lift, push, pick-and-place로 확장하였다. MuJoCo 환경에서 multi-task dataset을 생성하고 OpenVLA-7B 모델을 LoRA fine-tuning하였으며, 모델이 예측한 7D action을 RaccoonBot의 4DOF arm 구조에 맞게 변환하여 실행할 수 있도록 구성하였다.

이를 통해 자연어 instruction 기반 RaccoonBot manipulation pipeline을 구성하였고, MuJoCo inference 결과를 통해 fine-tuned OpenVLA 모델이 다양한 조작 task에 대해 target object motion을 생성할 수 있음을 확인하였다.
