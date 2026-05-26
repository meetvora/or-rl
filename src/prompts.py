from __future__ import annotations


GENERATION_INSTRUCTION = """
If the problem is a constrained optimization problem, solve it by writing executable Python code, using Google OR-Tools.

Output only:
<CODE>
# executable Python code that computes and prints the final answer
</CODE>
and terminate.

Do not include prose outside the <CODE> block in this case. 
Any brief reasoning should be written as Python comments inside the code.
User prefers short crisp code and so it is okay:
    1. to use short variable names
    2. to skip comments unless very necessary.
"""
