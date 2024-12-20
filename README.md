[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/gccUt70s)
# Final Project

Please follow the syllabus to submit a pdf project report in the form of a paper, along with runnable codes and README.md. 


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
    
