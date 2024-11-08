# Imports
import streamlit as st
import pandas as pd
from openai import OpenAI
import json

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)

# Streamlit app setup
st.title("Math Expression Generator")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head(10))  # Display first 10 rows

    # Annotate columns based on data type
    data.columns = [f"{col}(num)" if pd.api.types.is_numeric_dtype(data[col]) else f"{col}(text)" for col in data.columns]

    prompt_template = f"""
        Your task is to generate a mathematical expression that aligns with the user's intent for any dataset. Ensure the expression includes
        supported arithmetic operators (+, -, *, /) and any relevant functions [SUM(col), AVG(col), INT(col), ABS(col), MIN(col), MAX(col), STDDEV(col), VARIANCE(col), COUNT()].
        
        **Expression Guidelines**:
        - All the column names should be with (num) or (text) within quotes. Eg. "column1 (num)" or "column2 (text)"
        - If the user asks for one column, just give that column in the expression. And conditions can be applied on that one column
            Eg: Give the column1 where column3 contains 1
               Expected Output: {{
                  "Expression": ""column1 (num)"",
                  "Condition_Groups": [
                      {{"Column_Name": "column3 (num)", "Column_Operator": "contains", "Operand_Type": "Value", "Operand": [1]}}
                  ]
              }}
        - If the user asks for average of different columns (mean and average are same):
            Eg: Give the average/mean of column1, column2, column3.
               Expected Output: {{"Expression": "("column1 (num)" + "column2 (num)" + "column3 (num)") / 3", "Condition_Groups": [] }}
               
        - If the user asks the total/sum of different columns with different conditions:
            Eg. Give the total/sum of column1, column2, column3 where column4 < 10 and column5 has 'test'
               Expected Output: {{
                  "Expression": "("column1 (num)" + "column2 (num)" + "column3 (num)")",
                  "Condition_Groups": [
                      {{
                          "Group_Operator": "and",
                          "Conditions": [
                              {{"Column_Name": "column4 (num)", "Column_Operator": "is less than", "Operand_Type": "Value", "Operand": [10]}},
                              {{"Column_Name": "column5 (text)", "Column_Operator": "is one of", "Operand_Type": "Value", "Operand": ["test"]}}
                          ]
                      }}
                  ]
              }}
          
        - `INT(col)` rounds off the values and `ABS(col)` makes the values positive.
        - Use ONLY the FUNCTIONS listed above, and FUNCTIONS should be applied ONLY on one column.
        - `COUNT()` give the total row count of the dataset and it WILL NOT take any column (exception in functions)
        
        **Multiple Column Handling**:
          - If the user mentions more than one column in the request, use ONLY OPERATORS in the expression.
          - For example, if the user requests to find the average of two columns, ONLY USE OPERATORS.
        
        **Percentage Calculation**:
        - When calculating percentages between multiple columns without a user-specified total, assume the total to be 100 multiplied by the number of columns involved
        - Eg: For finding the percentage of values across ColumnA and ColumnB without a total provided (Remember to multiply by 100): ("ColumnA" + "ColumnB" / (100 * 2)) * 100
        
        **Percentage Calculation**:
        - The conditional operator (such as "and" or "or") is applied between two columns rather than within a single column.
        
        **Warnings**:
        - If the prompt requests applying a function (like max, int, abs, min, stddev, variance, count) across multiple columns, include a message:
          "Warning: Functions cannot be applied across multiple columns."
        - If the user makes an invalid or nonsensical request, respond with:
          "I don't understand. Please change your request."
        
        **Conditions**:
        - Include any specified conditions in the output.
        - For conditions, identify the target column, operator, operand type, and operand to ensure accurate filtering within the dataset.
        
        Examples:
        1. Give the column1 
           Expected Output: {{"Expression": ""column1 (num)"", "Condition_Groups": [] }}
           
        2. Give the percentage of column1
            Expected Output: {{"Expression": ""column1 (num)"/ 100 * 100", "Condition_Groups": [] }}
           
        3. Assuming the total (column1, column2, column3) is 500, give the percentage of those columns
            Expected Output: {{"Expression": "("column1 (num)" + "column2 (num)" + "column3 (num)") / 500 * 100", "Condition_Groups": [] }}

        3. Give the mean of column1 where column3 is less than 150.
          Expected Output: {{
              "Expression": "AVG("column1 (num)")",
              "Condition_Groups": [
                  {{"Column_Name": "column3 (num)", "Column_Operator": "is less than", "Operand_Type": "Value", "Operand": [150]}}
              ]
          }}

        4. Give the column1 where column2 >= column3
           Expected Output: {{
              "Expression": ""column1 (num)"",
              "Condition_Groups": [
                  {{"Column_Name": "column2 (num)", "Column_Operator": "is greater than or equal to", "Operand_Type": "Column Value", "Operand": ["column3 (num)"]}}
              ]
          }}
          
        5. Give the score range of column1 where column2 is null/empty
           Expected Output: {{
              "Expression": "MAX("column1 (num)") - MIN("column1 (num)")", 
              "Condition_Groups": [
                  {{"Column_Name": "column2 (num)", "Column_Operator": "is Empty", "Operand_Type": "Value", "Operand": ["Null"]}}
              ]
          }}

        6. Give the mean/average of column1 and column5 where column2 is greater than 50 and column3 contains 'pass'.
          Expected Output: {{
              "Expression": "("column1 (num)" + "column5 (num)") / 2",
              "Condition_Groups": [
                  {{
                      "Group_Operator": "and",
                      "Conditions": [
                          {{"Column_Name": "column2 (num)", "Column_Operator": "is greater than", "Operand_Type": "Value", "Operand": [50]}},
                          {{"Column_Name": "column3 (text)", "Column_Operator": "contains", "Operand_Type": "Value", "Operand": ["pass"]}}
                      ]
                  }}
              ]
          }}

        7. Give column1 where column2 has values 'pass' or 'fail' and column3 has values between 20 and 90
          Expected Output: {{
              "Expression": ""column1 (num)"",
              "Condition_Groups": [
                  {{
                      "Group_Operator": "and",
                      "Conditions": [
                          {{"Column_Name": "column2 (text)", "Column_Operator": "is one of", "Operand_Type": "Value", "Operand": ["pass", "fail"]}},
                          {{"Column_Name": "column3 (num)", "Column_Operator": "in between", "Operand_Type": "Value", "Operand": [20, 90]}}
                      ]
                  }}
              ]
          }}
    """

    # Step 2: Prompt Input
    user_prompt = st.text_area(
        "Enter your prompt (e.g., Give me average of 'Maths' column where 'result' column is 'pass'):")
    full_prompt = f"""
        {prompt_template}

        User Intent: {user_prompt}
        The columns of the dataset are as follows:
        """

    # Append column values to the prompt for context
    for col in data.columns:
        column_values = [str(row) for row in data[col].head(20)]
        all_values_str = ", ".join(column_values)
        full_prompt += f"\nColumn: {col}\nValues:\n" + all_values_str + "\n"
    full_prompt += "\nGiven the intent, output just the dictionary and no other text."

    # Define the JSON schema format for OpenAI response
    FORMAT = {
      "type": "json_schema",
      "json_schema": {
        "name": "conditional_expression",
        "schema": {
          "type": "object",
          "properties": {
            "Expression": {"type": "string"},
            "Condition_Groups": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "Group_Operator": {
                    "type": "string",
                    "enum": ["and", "or"]
                  },
                  "Conditions": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "Column_Name": {"type": "string"},
                        "Column_Operator": {
                          "type": "string",
                          "enum": [
                            "is",
                            "is one of",
                            "is NOT",
                            "is NOT one of",
                            "is less than",
                            "is less than or equal to",
                            "is greater than",
                            "is greater than or equal to",
                            "is the maximum value",
                            "is NOT the maximum value",
                            "is the minimum value",
                            "is NOT the minimum value",
                            "is Empty",
                            "is NOT Empty",
                            "in between",
                            "contains",
                            "does NOT contain",
                            "starts with",
                            "ends with",
                            "does NOT start with",
                            "does NOT end with",
                            "is earlier than",
                            "is on or earlier than",
                            "is later than",
                            "is on or later than"
                          ]
                        },
                        "Operand_Type": {
                          "type": "string",
                          "enum": ["Value", "Column Value"]
                        },
                        "Operand": {
                          "type": "array",
                          "items": {
                            "anyOf": [
                              {"type": "string"},
                              {"type": "number"},
                              {"type": "boolean"},
                              {"type": "null"}
                            ]
                          }
                        }
                      },
                      "required": ["Column_Name", "Column_Operator", "Operand_Type", "Operand"],
                      "additionalProperties": False
                    }
                  }
                },
                "required": ["Group_Operator", "Conditions"],
                "additionalProperties": False
              }
            }
          },
          "required": ["Expression", "Condition_Groups"],
          "additionalProperties": False
        },
        "strict": True
      }
    }


    # # Function to generate the dictionary based on the prompt
    # def generate_expression_dict(data, prompt):
    #     response = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[{"role": "system", "content": "You are a helpful assistant."},
    #                   {"role": "user", "content": prompt}],
    #         response_format=FORMAT,
    #         max_tokens=200,
    #         temperature=0
    #     )
    #     output = response.choices[0].message.content.strip()
    #     print(output)
    #     return json.loads(output)

    # Button to apply request and get output
    if st.button("Apply"):

        # Function to generate the dictionary based on the prompt
        def generate_expression_dict(prompt):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                response_format=FORMAT,
                max_tokens=200,
                temperature=0
            )
            output = response.choices[0].message.content.strip()
            print(output)
            return json.loads(output)

        # List of operators for which Operand Type and Operand should be hidden
        no_operand_operators = [
            "is Empty", "is NOT Empty", "is the maximum value",
            "is NOT the maximum value", "is the minimum value",
            "is NOT the minimum value"
        ]

        # Generate and display output
        try:
            output_dict = generate_expression_dict(full_prompt)

            # Check if the expression contains any warnings
            warning_messages = [
                "Warning: Functions cannot be applied across multiple columns",
                "I don't understand. Please change your request"
            ]

            # Display warnings if they exist and skip the expression display
            if any(warning in output_dict["Expression"] for warning in warning_messages):
                st.write("### Warning")
                for warning in warning_messages:
                    if warning in output_dict["Expression"]:
                        st.write(warning)
            elif not output_dict["Expression"] or "(text)" in output_dict["Expression"]:
                # Display a specific warning if the expression is empty
                st.write("### Warning")
                st.write(
                    "While the conditions can be on any column, the base column (expression) cannot be a text/date column")
            else:
                # Display the expression if no warnings are present
                st.write("### Generated Expression")
                st.text(output_dict["Expression"].replace("'", '"'))

                # Display conditions if any
                if output_dict.get("Condition_Groups"):
                    st.write("### Conditions")
                    for i, group in enumerate(output_dict["Condition_Groups"]):
                        # Only display group operator if there are multiple conditions
                        if len(group["Conditions"]) > 1:
                            st.write(f"**Group Operator**: {group['Group_Operator']}")
                        for j, condition in enumerate(group["Conditions"]):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.selectbox("Column Name", [condition["Column_Name"]], key=f"column_name_{i}_{j}")
                            with col2:
                                st.selectbox("Operator", [condition["Column_Operator"]], key=f"operator_{i}_{j}")

                            # Only display Operand Type and Operand if the operator is not in the no_operand_operators list
                            if condition["Column_Operator"] not in no_operand_operators:
                                with col3:
                                    st.selectbox("Operand Type", [condition["Operand_Type"]],
                                                 key=f"operand_type_{i}_{j}")
                                with col4:
                                    st.text_input("Operand", ", ".join(map(str, condition["Operand"])),
                                                  key=f"operand_{i}_{j}")
        except json.JSONDecodeError:
            st.write("An error occurred while processing your request. Please try again.")
