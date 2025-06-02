# Communicative Watch-And-Help

The CoTS Playground — a friendly environment where intelligent agents collaborate, communicate, and conquer household tasks together!  
Based on the C-WAH environment initially adapted by [CoELA](https://github.com/UMass-Embodied-AGI/CoELA/tree/master/cwah), this project further extends and customizes it to support our proposed CoTS framework.

## Setup
### Step 1: Download VirtualHome Simulator and API
We follow the exact setup steps from the original C-WAH environment.

Clone the [VirtualHome API](https://github.com/xavierpuigf/virtualhome.git) repository:

```bash
git clone --branch wah https://github.com/xavierpuigf/virtualhome.git
```

Download the [Simulator](https://drive.google.com/file/d/1L79SxE07Jt-8-_uCvNnkwz5Kf6AjtaGp/view?usp=sharing) (Linux x86-64 version), and unzip it.

```bash
gdown https://drive.google.com/uc?id=1L79SxE07Jt-8-_uCvNnkwz5Kf6AjtaGp
unzip executable.zip
chmod +x executable/linux_exec.v2.3.0.x86_64
```

Your folder structure should look like this:

```bash
|--cwah/
|--virtualhome/
|--executable/
```

### Step 2: Environment Setup

```bash
cd cwah
conda create --name cwah python=3.8
conda activate cwah
pip install -r requirements.txt
pip install -U langgraph
```

## Run Experiments

We’ve provided sample scripts.

To get started with symbolic observations and two LLM agents:

```bash
./scripts/symbolic_obs_llm_llm.sh
```

## Environment Details

This project operates in the Communicative Watch-And-Help (CWAH) environment, an extension of the Watch-And-Help challenge, which allows agents to send natural-language messages to one another.

### Tasks 

Five types of tasks are available in C-WAH, named `Prepare afternoon tea`, `Wash dishes`, `Prepare a meal`, `Put groceries`, and `Set up a dinner table`. These tasks include a range of housework, and each task contains a few subgoals, which are described by predicates. A predicate is in `ON/IN(x, y)` format, that is, `Put x ON/IN y`. The detailed descriptions of tasks are listed in the following table:

| Task Name | Predicate Set |
| ------- | ------- |
| Prepare afternoon tea   | ON(cupcake,coffeetable), ON(pudding,coffeetable), ON(apple,coffeetable), ON(juice,coffeetable), ON(wine,coffeetable)  |
| Wash dishes  | IN(plate,dishwasher), IN(fork,dishwasher)  |
| Prepare a meal | ON(coffeepot,dinnertable),ON(cupcake,dinnertable), ON(pancake,dinnertable), ON(poundcake,dinnertable), ON(pudding,dinnertable), ON(apple,dinnertable), ON(juice,dinnertable), ON(wine,dinnertable) |
|Put groceries | IN(cupcake,fridge), IN(pancake,fridge), IN(poundcake,fridge), IN(pudding,fridge), IN(apple,fridge), IN(juice,fridge), IN(wine,fridge) |
|Set up a dinner table | ON(plate,dinnertable), ON(fork,dinnertable) |

The task goal is to satisfy all the given subgoals within $250$ time steps, and the number of subgoals in each task ranges from $3$ to $5$. 

### Metrics

  - **Average Steps (L)**: Number of steps to finish the task;
  - **Efficiency Improvement (EI)**: The efficiency improvements of cooperating with base agents.
