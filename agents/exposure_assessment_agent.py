"""Exposure assessment agent for controlling exposure analysis"""
import json
from langchain_core.messages import AIMessage
from state import AgentState


class ExposureAssessmentAgent:
    """Handles exposure assessment control requests"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def process(self, state: AgentState) -> AgentState:
        """Process exposure assessment requests using LLM-based intent parsing"""
        messages = state["messages"]
        last_message = messages[-1].content
        uploaded_files = state.get("uploaded_files", [])
        
        print(f"Exposure Assessment Agent processing: {last_message}")
        print(f"Available uploaded files: {uploaded_files}")
        
        # Check if this is a response to a previous clarification
        pending_action = state.get("pending_action")
        if pending_action and pending_action.get("suggested_action", {}).get("tool") == "run_exposure_analysis":
            print("Processing clarification response for exposure assessment")
            suggested = pending_action.get("suggested_action", {})
            awaiting = suggested.get("awaiting")
            
            # Parse user's selection (e.g., "1", "2", "1 and 3", etc.)
            try:
                # Extract numbers from response
                import re
                numbers = re.findall(r'\d+', last_message.lower())
                
                if awaiting == "hazard_source_selection":
                    # User is choosing between existing or imported hazard data
                    # Accept numbers OR keywords
                    msg_lower = last_message.lower()
                    choice = None
                    
                    if numbers:
                        choice = int(numbers[0])
                    elif "existing" in msg_lower or "system" in msg_lower:
                        choice = 1
                    elif "import" in msg_lower or "upload" in msg_lower or "file" in msg_lower:
                        choice = 2
                    
                    if not choice:
                        response_data = {
                            "text": "Please select an option:\n• Reply '1' or 'existing' for existing data\n• Reply '2' or 'imported' for imported data"
                        }
                        json_response = json.dumps(response_data)
                        return {"messages": state["messages"] + [AIMessage(content=json_response)]}
                    if choice == 1:
                        # User chose existing data for hazard
                        response_data = {
                            "type": "clarification",
                            "question": "✅ **Hazard data source selected: Existing data**\n\nNow, which **exposure elements** would you like to use?",
                            "options": [
                                "Option 1: Use existing data (from the system)",
                                "Option 2: Use imported data (files you've uploaded)"
                            ],
                            "suggested_action": {
                                "tool": "run_exposure_analysis",
                                "action": "run",
                                "hazard_source": "existing",
                                "awaiting": "element_source_selection"
                            }
                        }
                    elif choice == 2:
                        # User chose imported data for hazard
                        if not uploaded_files:
                            response_data = {
                                "text": "❌ No uploaded files detected. Please upload hazard data files first."
                            }
                        else:
                            response_data = {
                                "type": "clarification",
                                "question": "Perfect! Please select which uploaded file(s) to use as **hazard data**:\n\n*Reply with a number or file name (e.g., '1' or 'Flood-25Year-Apayao.geojson')*",
                                "options": [f"Option {i+1}: {file}" for i, file in enumerate(uploaded_files)],
                                "suggested_action": {
                                    "tool": "run_exposure_analysis",
                                    "action": "run",
                                    "hazard_source": "imported",
                                    "awaiting": "hazard_selection_step"
                                }
                            }
                    else:
                        response_data = {
                            "text": "Please select either 1 or 2."
                        }
                    
                    json_response = json.dumps(response_data)
                    return {"messages": state["messages"] + [AIMessage(content=json_response)]}
                
                elif awaiting == "element_source_selection":
                    # User is choosing between existing or imported element data
                    # Accept numbers OR keywords
                    msg_lower = last_message.lower()
                    choice = None
                    
                    if numbers:
                        choice = int(numbers[0])
                    elif "existing" in msg_lower or "system" in msg_lower:
                        choice = 1
                    elif "import" in msg_lower or "upload" in msg_lower or "file" in msg_lower:
                        choice = 2
                    
                    if not choice:
                        response_data = {
                            "text": "Please select an option:\n• Reply '1' or 'existing' for existing data\n• Reply '2' or 'imported' for imported data"
                        }
                        json_response = json.dumps(response_data)
                        return {"messages": state["messages"] + [AIMessage(content=json_response)]}
                    hazard_source = suggested.get("hazard_source", "existing")
                    hazard_data = suggested.get("hazard_data", [])
                    
                    if choice == 1:
                        # User chose existing data for elements
                        response_data = {
                            "text": "Excellent! Now, which **existing exposure elements** would you like to analyze?\n\nAvailable options:\n• Land Cover\n• Transportation Networks (Roads)\n• Point Features (Buildings, facilities)\n\nYou can say something like:\n• \"Use land cover\"\n• \"Analyze roads and buildings\"\n• \"Use all available elements\""
                        }
                    elif choice == 2:
                        # User chose imported data for elements
                        if not uploaded_files:
                            response_data = {
                                "text": "❌ No uploaded files detected. Please upload exposure element files first."
                            }
                        else:
                            response_data = {
                                "type": "clarification",
                                "question": "Perfect! Please select which uploaded file(s) to use as **exposure elements**:\n\n*Reply with a number or file name (e.g., '2' or 'Landcover-Apayao.geojson')*",
                                "options": [f"Option {i+1}: {file}" for i, file in enumerate(uploaded_files)],
                                "suggested_action": {
                                    "tool": "run_exposure_analysis",
                                    "action": "run",
                                    "hazard_source": hazard_source,
                                    "hazard_data": hazard_data,
                                    "element_source": "imported",
                                    "awaiting": "element_selection"
                                }
                            }
                    else:
                        response_data = {
                            "text": "Please select either 1 or 2."
                        }
                    
                    json_response = json.dumps(response_data)
                    new_messages = state["messages"] + [AIMessage(content=json_response)]
                    print(f"DEBUG: element_source_selection - appended message, total messages: {len(new_messages)}")
                    return {"messages": new_messages}
                
                # IMPORTANT: Check for confirm_analysis FIRST before trying to parse as file selection
                elif awaiting == "confirm_analysis":
                    # User is confirming whether to run the analysis
                    # Accept numbers OR keywords
                    msg_lower = last_message.lower()
                    choice = None
                    
                    if numbers:
                        choice = int(numbers[0])
                    elif any(word in msg_lower for word in ["yes", "run", "start", "go", "proceed", "confirm"]):
                        choice = 1
                    elif any(word in msg_lower for word in ["no", "cancel", "stop", "abort"]):
                        choice = 2
                    
                    if choice is None:
                        response_data = {
                            "text": "Please confirm:\n• Reply '1' or 'yes' to run the analysis\n• Reply '2' or 'no' to cancel"
                        }
                        json_response = json.dumps(response_data)
                        return {"messages": state["messages"] + [AIMessage(content=json_response)]}
                    
                    if choice == 1:
                        # User confirmed, run analysis
                        response_data = {
                            "tool": "run_exposure_analysis",
                            "action": "run",
                            "hazard_source": suggested.get("hazard_source", "existing"),
                            "hazard_data": suggested.get("hazard_data", []),
                            "element_source": suggested.get("element_source", "existing"),
                            "element_data": suggested.get("element_data", []),
                            "requires_frontend": True,
                            "message": "🔄 **Starting exposure analysis...**\n\nThis may take a while depending on the size of your data. Please wait for the analysis to complete."
                        }
                    else:
                        # User cancelled
                        response_data = {
                            "text": "Analysis cancelled. You can start over by saying 'Do exposure assessment' or specify your data directly."
                        }
                    
                    json_response = json.dumps(response_data)
                    # Clear pending action after processing
                    return {"messages": state["messages"] + [AIMessage(content=json_response)], "pending_action": None}
                
                # Try to parse as numbers first (for file selection)
                selected_files = []
                if numbers:
                    selected_indices = [int(n) - 1 for n in numbers]  # Convert to 0-based index
                    selected_files = [uploaded_files[i] for i in selected_indices if i < len(uploaded_files)]
                
                # If no numbers, try to match file names
                if not selected_files:
                    for file in uploaded_files:
                        if file.lower() in last_message.lower() or last_message.lower() in file.lower():
                            selected_files.append(file)
                
                if not selected_files:
                    # No valid selection, ask again
                    response_data = {
                        "text": "Please select a file:\n• Reply with a number (e.g., '1')\n• Or reply with the file name (e.g., 'Flood-25Year-Apayao.geojson')\n• Or say 'cancel' to stop"
                    }
                    json_response = json.dumps(response_data)
                    return {"messages": state["messages"] + [AIMessage(content=json_response)]}
                
                if not selected_files:
                    # Invalid selection
                    response_data = {
                        "text": f"Invalid selection. Please select a number between 1 and {len(uploaded_files)}."
                    }
                    json_response = json.dumps(response_data)
                    state["messages"].append(AIMessage(content=json_response))
                    return state
                
                if awaiting == "hazard_selection_step":
                    # User selected hazard files in step-by-step mode, now ask for element source
                    response_data = {
                        "type": "clarification",
                        "question": f"✅ **Hazard file selected:** {', '.join(selected_files)}\n\nNow, which **exposure elements** would you like to use?\n\n*Reply with a number or keywords (e.g., '1', 'existing', 'imported')*",
                        "options": [
                            "Option 1: Use existing data (from the system)",
                            "Option 2: Use imported data (files you've uploaded)"
                        ],
                        "suggested_action": {
                            "tool": "run_exposure_analysis",
                            "action": "run",
                            "hazard_source": "imported",
                            "hazard_data": selected_files,
                            "awaiting": "element_source_selection"
                        }
                    }
                    json_response = json.dumps(response_data)
                    state["messages"].append(AIMessage(content=json_response))
                    return state
                
                elif awaiting == "hazard_selection":
                    # User selected hazard files, now check if we need element selection
                    if suggested.get("element_source") == "imported":
                        # Need to ask for element selection next
                        response_data = {
                            "type": "clarification",
                            "question": "Great! Now please select which uploaded file(s) to use as **exposure elements**:\n\n*Reply with a number or file name (e.g., '2' or 'Landcover-Apayao.geojson')*",
                            "options": [f"Option {i+1}: {file}" for i, file in enumerate(uploaded_files)],
                            "suggested_action": {
                                "tool": "run_exposure_analysis",
                                "action": "run",
                                "hazard_source": "imported",
                                "hazard_data": selected_files,
                                "element_source": "imported",
                                "awaiting": "element_selection"
                            }
                        }
                    else:
                        # Check if element data is specified
                        element_source = suggested.get("element_source", "existing")
                        element_data = suggested.get("element_data", [])
                        
                        if element_source == "existing" and len(element_data) == 0:
                            # Need to ask user which existing elements to use
                            response_data = {
                                "type": "clarification",
                                "question": "✅ **Hazard data selected!**\n\nNow, which **exposure elements** would you like to analyze?\n\nAvailable options:\n• Land Cover\n• Transportation Networks (Roads)\n• Point Features (Buildings, facilities)\n\nYou can say something like:\n• \"Use land cover\"\n• \"Analyze roads and buildings\"\n• \"Use all available elements\"",
                                "suggested_action": {
                                    "tool": "run_exposure_analysis",
                                    "action": "run",
                                    "hazard_source": "imported",
                                    "hazard_data": selected_files,
                                    "element_source": "existing",
                                    "awaiting": "element_data_selection"
                                }
                            }
                        else:
                            # Ask for confirmation before running
                            hazard_display = ", ".join(selected_files)
                            element_display = ", ".join(element_data) if element_data else "existing data"
                            
                            response_data = {
                                "type": "clarification",
                                "question": f"✅ **All data selected!**\n\n**Hazard:** {hazard_display}\n**Elements:** {element_display}\n\nReady to run the exposure analysis?\n\n*Reply '1'/'yes' to run or '2'/'no' to cancel*",
                                "options": [
                                    "Option 1: Yes, run the analysis",
                                    "Option 2: No, cancel"
                                ],
                                "suggested_action": {
                                    "tool": "run_exposure_analysis",
                                    "action": "run",
                                    "hazard_source": "imported",
                                    "hazard_data": selected_files,
                                    "element_source": element_source,
                                    "element_data": element_data,
                                    "awaiting": "confirm_analysis"
                                }
                            }
                elif awaiting == "element_selection":
                    # User selected element files, ask for confirmation before running
                    print(f"DEBUG: element_selection case - selected_files: {selected_files}")
                    print(f"DEBUG: hazard_source: {suggested.get('hazard_source')}, hazard_data: {suggested.get('hazard_data')}")
                    response_data = {
                        "type": "clarification",
                        "question": "✅ **All data selected!**\n\n**Hazard:** " + ", ".join(suggested.get("hazard_data", [])) + "\n**Elements:** " + ", ".join(selected_files) + "\n\nReady to run the exposure analysis?\n\n*Reply '1'/'yes' to run or '2'/'no' to cancel*",
                        "options": [
                            "Option 1: Yes, run the analysis",
                            "Option 2: No, cancel"
                        ],
                        "suggested_action": {
                            "tool": "run_exposure_analysis",
                            "action": "run",
                            "hazard_source": suggested.get("hazard_source", "existing"),
                            "hazard_data": suggested.get("hazard_data", []),
                            "element_source": "imported",
                            "element_data": selected_files,
                            "awaiting": "confirm_analysis"
                        }
                    }
                    print(f"DEBUG: response_data set for confirmation: {response_data}")
                
                elif awaiting == "element_data_selection":
                    # User specified which existing elements to use (e.g., "Use land cover")
                    # Parse the user's response using LLM
                    element_prompt = f"""Parse this exposure element selection request.

User request: {last_message}

Extract which elements the user wants to analyze. Available options:
- Land Cover
- Transportation Networks (Roads)
- Point Features (Buildings, facilities)

Respond with a JSON array of element names. Examples:
- "Use land cover" → ["Land Cover"]
- "Analyze roads and buildings" → ["Transportation Networks", "Point Features"]
- "Use all" → ["Land Cover", "Transportation Networks", "Point Features"]

Respond ONLY with valid JSON array: ["element1", "element2"]
"""
                    try:
                        llm_response = self.llm.invoke(element_prompt, system_prompt="You are a JSON parser. Respond only with a JSON array.")
                        cleaned = llm_response.strip()
                        if cleaned.startswith("```json"):
                            cleaned = cleaned[7:]
                        if cleaned.startswith("```"):
                            cleaned = cleaned[3:]
                        if cleaned.endswith("```"):
                            cleaned = cleaned[:-3]
                        cleaned = cleaned.strip()
                        
                        element_list = json.loads(cleaned)
                        
                        # Ask for confirmation before running
                        hazard_display = ", ".join(suggested.get("hazard_data", [])) if suggested.get("hazard_data") else "existing data"
                        element_display = ", ".join(element_list)
                        
                        response_data = {
                            "type": "clarification",
                            "question": f"✅ **All data selected!**\n\n**Hazard:** {hazard_display}\n**Elements:** {element_display}\n\nReady to run the exposure analysis?\n\n*Reply '1'/'yes' to run or '2'/'no' to cancel*",
                            "options": [
                                "Option 1: Yes, run the analysis",
                                "Option 2: No, cancel"
                            ],
                            "suggested_action": {
                                "tool": "run_exposure_analysis",
                                "action": "run",
                                "hazard_source": suggested.get("hazard_source", "imported"),
                                "hazard_data": suggested.get("hazard_data", []),
                                "element_source": "existing",
                                "element_data": element_list,
                                "awaiting": "confirm_analysis"
                            }
                        }
                    except Exception as e:
                        print(f"Error parsing element selection: {e}")
                        response_data = {
                            "text": "I couldn't understand which elements you want to analyze. Please try again:\n• \"Use land cover\"\n• \"Analyze roads\"\n• \"Use all available elements\""
                        }
                else:
                    response_data = {
                        "text": "I'm not sure what selection you're making. Please try again."
                    }
                
                json_response = json.dumps(response_data)
                new_messages = state["messages"] + [AIMessage(content=json_response)]
                print(f"DEBUG: Appended message at end of clarification handling, total messages: {len(new_messages)}, awaiting was: {awaiting}")
                return {"messages": new_messages}
                
            except Exception as e:
                print(f"Error processing clarification response: {e}")
                response_data = {
                    "text": "I couldn't understand your selection. Please select a file by number (e.g., '1' or '2')."
                }
                json_response = json.dumps(response_data)
                state["messages"].append(AIMessage(content=json_response))
                return state
        
        # Use LLM to parse intent
        intent_prompt = f"""Parse this exposure assessment request.

User request: {last_message}

TASK: Extract these fields ONLY:
1. action: "run_analysis" or "clear_steps" or "select_hazard" or "select_element"
2. hazard_source: "existing" or "imported"
3. hazard_data: [] (ALWAYS EMPTY unless user names specific files)
4. element_source: "existing" or "imported"
5. element_data: [] (ALWAYS EMPTY unless user names specific files)

RULES:
- If user says "with imported files/data" WITHOUT naming files → hazard_source="imported", hazard_data=[], element_source="imported", element_data=[]
- If user says "Perform an exposure assessment with imported files" → hazard_source="imported", hazard_data=[], element_source="imported", element_data=[]
- If user names a file like "use Flood-25Year.geojson" → include it in hazard_data or element_data
- Default: hazard_source="existing", hazard_data=[], element_source="existing", element_data=[]

IGNORE THIS (for reference only): Available files are {uploaded_files}

Respond with JSON only:
{{"action": "run_analysis", "hazard_source": "imported", "hazard_data": [], "element_source": "imported", "element_data": []}}"""
        
        try:
            llm_response = self.llm.invoke(intent_prompt, system_prompt="You are a JSON parser for exposure assessment commands. Respond only with valid JSON.")
            print(f"LLM parsed intent: {llm_response}")
            
            # Clean up response - remove markdown code blocks if present
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            parsed = json.loads(cleaned_response)
            action = parsed.get("action", "none")
            
            print(f"Parsed action: {action}")
            
            if action == "none" or not action:
                # Could not parse intent
                response_data = {
                    "text": "I can help you with exposure assessment. You can:\n\n• Run exposure analysis\n• Select hazard data (flood, earthquake, etc.)\n• Select exposure elements (land cover, roads, buildings)\n• Clear assessment steps\n\nWhat would you like to do?"
                }
            elif action == "run_analysis":
                hazard_source = parsed.get("hazard_source", "existing")
                element_source = parsed.get("element_source", "existing")
                hazard_data_list = parsed.get("hazard_data", [])
                element_data_list = parsed.get("element_data", [])
                
                # Check if user hasn't specified any data sources (step-by-step guidance)
                if (hazard_source == "existing" and not hazard_data_list and 
                    element_source == "existing" and not element_data_list):
                    # Guide user to choose data source type first
                    response_data = {
                        "type": "clarification",
                        "question": "Let's set up your exposure assessment! First, which **hazard data** would you like to use?\n\n*Reply with a number or keywords (e.g., '1', 'existing', 'imported')*",
                        "options": [
                            "Option 1: Use existing data (from the system)",
                            "Option 2: Use imported data (files you've uploaded)"
                        ],
                        "suggested_action": {
                            "tool": "run_exposure_analysis",
                            "action": "run",
                            "awaiting": "hazard_source_selection"
                        }
                    }
                # Check if imported data is requested but no files specified
                elif hazard_source == "imported" and not hazard_data_list:
                    if not uploaded_files:
                        response_data = {
                            "text": "❌ No uploaded files detected. Please upload hazard data files first before running exposure analysis with imported data."
                        }
                    else:
                        # Ask user to select from uploaded files
                        response_data = {
                            "type": "clarification",
                            "question": "Please select which uploaded file(s) to use as **hazard data**:\n\n*Reply with a number or file name (e.g., '1' or 'Flood-25Year-Apayao.geojson')*",
                            "options": [f"Option {i+1}: {file}" for i, file in enumerate(uploaded_files)],
                            "suggested_action": {
                                "tool": "run_exposure_analysis",
                                "action": "run",
                                "hazard_source": "imported",
                                "element_source": element_source,
                                "awaiting": "hazard_selection"
                            }
                        }
                elif element_source == "imported" and not parsed.get("element_data"):
                    if not uploaded_files:
                        response_data = {
                            "text": "❌ No uploaded files detected. Please upload exposure element files first before running exposure analysis with imported data."
                        }
                    else:
                        # Ask user to select from uploaded files
                        response_data = {
                            "type": "clarification",
                            "question": "Please select which uploaded file(s) to use as **exposure elements**:\n\n*Reply with a number or file name (e.g., '2' or 'Landcover-Apayao.geojson')*",
                            "options": [f"Option {i+1}: {file}" for i, file in enumerate(uploaded_files)],
                            "suggested_action": {
                                "tool": "run_exposure_analysis",
                                "action": "run",
                                "hazard_source": hazard_source,
                                "hazard_data": parsed.get("hazard_data", []),
                                "element_source": "imported",
                                "awaiting": "element_selection"
                            }
                        }
                else:
                    # Check if element data is also specified
                    element_data_list = parsed.get("element_data", [])
                    hazard_data_list = parsed.get("hazard_data", [])
                    
                    # Validate that specified imported files exist
                    if hazard_source == "imported" and hazard_data_list:
                        invalid_files = [f for f in hazard_data_list if f not in uploaded_files]
                        if invalid_files:
                            response_data = {
                                "text": f"❌ The following hazard files were not found in uploaded files: {', '.join(invalid_files)}\n\nAvailable files: {', '.join(uploaded_files)}"
                            }
                            json_response = json.dumps(response_data)
                            state["messages"].append(AIMessage(content=json_response))
                            return state
                    
                    if element_source == "imported" and element_data_list:
                        invalid_files = [f for f in element_data_list if f not in uploaded_files]
                        if invalid_files:
                            response_data = {
                                "text": f"❌ The following element files were not found in uploaded files: {', '.join(invalid_files)}\n\nAvailable files: {', '.join(uploaded_files)}"
                            }
                            json_response = json.dumps(response_data)
                            state["messages"].append(AIMessage(content=json_response))
                            return state
                    
                    if element_source == "existing" and len(element_data_list) == 0:
                        # Need to ask user which existing elements to use
                        response_data = {
                            "text": "✅ **Hazard data selected!**\n\nNow, which **exposure elements** would you like to analyze?\n\nAvailable options:\n• Land Cover\n• Transportation Networks (Roads)\n• Point Features (Buildings, facilities)\n\nYou can say something like:\n• \"Use land cover\"\n• \"Analyze roads and buildings\"\n• \"Use all available elements\""
                        }
                    else:
                        # All data specified, ask for confirmation
                        hazard_display = ", ".join(hazard_data_list) if hazard_data_list else "existing data"
                        element_display = ", ".join(element_data_list) if element_data_list else "existing data"
                        
                        response_data = {
                            "type": "clarification",
                            "question": f"✅ **All data selected!**\n\n**Hazard:** {hazard_display}\n**Elements:** {element_display}\n\nReady to run the exposure analysis?\n\n*Reply '1'/'yes' to run or '2'/'no' to cancel*",
                            "options": [
                                "Option 1: Yes, run the analysis",
                                "Option 2: No, cancel"
                            ],
                            "suggested_action": {
                                "tool": "run_exposure_analysis",
                                "action": "run",
                                "hazard_source": hazard_source,
                                "hazard_data": hazard_data_list,
                                "element_source": element_source,
                                "element_data": element_data_list,
                                "awaiting": "confirm_analysis"
                            }
                        }
            elif action == "clear_steps":
                # Clear assessment steps
                response_data = {
                    "tool": "control_exposure_assessment",
                    "action": "clear",
                    "requires_frontend": True
                }
            elif action == "select_hazard":
                # Select hazard data
                response_data = {
                    "tool": "control_exposure_assessment",
                    "action": "select_hazard",
                    "hazard_source": parsed.get("hazard_source", "existing"),
                    "hazard_data": parsed.get("hazard_data", []),
                    "requires_frontend": True
                }
            elif action == "select_element":
                # Select exposure elements
                response_data = {
                    "tool": "control_exposure_assessment",
                    "action": "select_element",
                    "element_source": parsed.get("element_source", "existing"),
                    "element_data": parsed.get("element_data", []),
                    "requires_frontend": True
                }
            else:
                response_data = {
                    "text": f"I'm not sure how to handle the action: {action}. Please try rephrasing your request."
                }
            
            json_response = json.dumps(response_data)
            state["messages"].append(AIMessage(content=json_response))
            
        except Exception as e:
            print(f"Error parsing exposure intent: {e}")
            # Fallback response
            fallback_response = json.dumps({
                "text": "I can help you with exposure assessment. Please specify what you'd like to do:\n\n• Run exposure analysis\n• Select hazard data\n• Select exposure elements\n• Clear steps"
            })
            state["messages"].append(AIMessage(content=fallback_response))
        
        return state
