# Homework5: Acceleration Techniques for Sampling in Diffusion Models

Homework 5 corresponds to Chapter 7 of the course.

## Objective
Explore and develop sampling strategies for diffusion models to improve the efficiency of image generation. Implement existing methods and propose an innovative strategy to enhance the performance.

## Problem Definition

### Problem 1(a): Detailed Analysis of Established Sampling Strategies
- **Task**: Examine at least two established sampling strategies. They could be but not limited to, e.g., Denoising Diffusion Implicit Models (DDIM) and Progressive Distillation. Provide a detailed analysis of their implementation.
- **Deliverables**:
  - **Mathematical Formulation**: Clearly present the mathematical formulation that define each strategy. You can upload pictures of equations but should clearly explain each notation.
  - **Operational Explanation**: Explain why these strategies accelerate the sampling process while maintaining the quality of generated images.

### Problem 1(b): Implementation and Performance Demonstration
- **Task**: Implement the sampling strategies studied in Problem 1(a) using a standard diffusion model (e.g., the model we trained on Butterflies or `google/ddpm-celebahq-25`).
- **Deliverables**:
  - **Performance Analysis**:
    - **Efficiency Improvement**: Demonstrate the improvement in sampling speed and reductions in computational resources.
    - **Quality Maintenance**: Evaluate and demonstrate how image quality is maintained using appropriate metrics, such as FID.

### Problem 1(c): Innovation in Sampling Strategies
- **Task**: Propose a novel sampling strategy that could potentially accelarate the diffusion process.
- **Deliverables**:
  - **Strategy Proposal**: Describe your new sampling strategy and explain its potential to enhance performance and efficiency.
  - **Performance Demonstration**: Implement your proposed strategy and compare its performance against established strategies.




## Rubric
### Clear problem statement (10%)

- Introduce the science domain in sufficient detail so others can understand the context.
- Specify the data source (which must be open-source to ensure reproducibility).
- Clearly define the goal or success criteria for the problem.


### Importance of the problem (10%)

- Explain the strong motivation behind solving the problem.
- Discuss the practical challenges that the solution will address.


### Existing literature and challenges to address (10%)

- Summarize approaches (with references) that have been developed to tackle this problem.
- Identify gaps or limitations in these approaches that your work aims to address.
- (Optional) Discuss why AI tools like ChatGPT or other out-of-the-box packages may not be sufficient to solve this problem.


### Evaluation criteria (10%)

- Specify the metrics for success (e.g., accuracy, ROC curve, computation time, memory usage).
- Provide a performance baseline from existing methods or straightforward approaches.

 
### Detailed explanation of the proposed approach and baseline methods (10%)

- Clearly explain the novelty of your approach (e.g., new perspective, new data processing, new method).
- Provide details on the baseline methods for experimental comparison.

### Primary experiments and additional ablation studies (10%)

- Implement the proposed approach and explain it improves upon baselines
- Conduct further ablation studies to investigate the effects of various hyperparameters, dataset sizes, or specific algorithmic components in your proposed approach. These can be used to write insights and limitations of your approach (see below).


### Write-up that summarizes the main results and findings (10%)

- Provide a clear, concise summary of your problem, solution, and results, in a Markdown file called Homework1-[TeamName].md or a Jupyter notebook named Homework1-[TeamName].ipynb.
- Discuss the results from primary experiments and ablation studies


### Limitations of your submitted work (10%)

- Identify and explain the limitations of your work.
- Avoid vague or generic statements; use specific examples or experimental results to support your claims.


### Critical thinking on future work (10%)

- Suggest new research directions that address the limitations you discussed.
- Be specific and try to propose innovative or ambitious ideas even if they are not immediately doable.


### Git repo with clear code and documentation (10%)

- Ensure that the submitted code is well-commented, organized, and executable.
- Provide instructions for running your code (e.g., by executing main.py in a Python 3.8 environment, with all dependencies listed in requirements.txt).
- The repo should include a Markdown file titled Homework1-[TeamName].md or a Jupyter notebook that connects your codes with writeup.


## Additional Notes

- It is allowed to use AI tools like ChatGPT for any part of your homework. However, the final results should reflect your own independent thinking.
- In the writeup, be concise, informative, and to the point. Reduce overly elaborate or verbose writing styles as often seen in AI-generated text.
- Please use Python 3.8 as the default Python version for grading purposes. If there is no guidance on how to reproduce the code (including data, dependent packages, and environment setup), or if the code is not directly executable to reproduce the reported results, the homework will receive a zero grade. So please do a sanity check before the submission deadline.
- Please ensure that your submission does not include any proprietary information, require third-party consent, or violate any confidentiality policies. All data and information used in the homework should be publicly available.
- Google Drive provides enough free space for large file transfers. When you need to submit large files (such as a trained model of a large size), follow these steps to ensure proper submission:
  - Use this command to create a zip file with your custom name: `zip my_large_file.zip my_large_file.extension`
  - Log into your Google Drive account and upload the zip file to your Drive under your course-specific folder.
  - Share the File: set the permission to Anyone with the link can View, and copy the File ID from the shareable link. It will look something like this: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
  - Provide the File ID in your submission and include in your write how to load and use the unzipped file once we download. We will use the same function `download_and_unzip()` used in [Chapter 2](https://genai-course.jding.org/en/latest/llm/index.html) to download and unzip.
  - Dryrun the above in your local computer to make sure the file is accessible.
    
