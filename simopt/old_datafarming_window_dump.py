# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:34:31 2024

@author: Owner
"""

class Solver_Datafarming_Window(tk.Toplevel):
    def __init__(self, master):
        
        self.master = master
        self.master.grid_rowconfigure(0, weight = 0)
        self.master.grid_rowconfigure(1, weight = 0)
        self.master.grid_rowconfigure(2, weight = 1)
        self.master.grid_rowconfigure(3, weight = 1)
        self.master.grid_rowconfigure(4, weight = 0)
        self.master.grid_rowconfigure(5, weight = 1)
        self.master.grid_rowconfigure(6, weight = 0)
        self.master.grid_rowconfigure(7, weight = 1)
        self.master.grid_rowconfigure(8, weight = 1)
        self.master.grid_rowconfigure(9, weight = 1)
        self.master.grid_rowconfigure(10, weight = 1)
        self.master.grid_columnconfigure(0, weight = 1)
        
        #Initalize frames
        self.title_frame = tk.Frame(master = self.master)
        self.title_frame.grid(row = 0, column = 0)
        self.title_frame.grid_rowconfigure(0, weight = 0)
        self.title_frame.grid_columnconfigure(0, weight = 1)
        
        self.solver_select_frame = tk.Frame(master = self.master)
        self.solver_select_frame.grid(row = 1, column =0)
        self.solver_select_frame.grid_rowconfigure(0, weight = 0)
        self.solver_select_frame.grid_columnconfigure(0, weight = 1)
        self.solver_select_frame.grid_columnconfigure(1, weight = 1)
        self.solver_select_frame.grid_columnconfigure(2, weight = 1)
        self.solver_select_frame.grid_columnconfigure(3, weight = 1)
        
        self.solver_frame = tk.Frame(master = self.master)
        self.solver_frame.grid(row = 2, column = 0)
        
        # frames created here so previous selections can be cleared
        self.design_frame = tk.Frame(master = self.master)
        self.problem_model_factors_frame = tk.Frame(master = self.master)
        self.problem_select_frame = tk.Frame(master = self.master)
        self.experiment_frame = tk.Frame(master = self.master)
        self.problem_frame = tk.Frame(master = self.problem_model_factors_frame)
        self.model_frame = tk.Frame(master = self.problem_model_factors_frame)
        self.notebook_frame = tk.Frame (master = self.master)
        self.design_view_frame = tk.Frame(master = self.master)
        self.create_pair_frame = tk.Frame(master = self.master)
        
        
        
        
        # Window title 
        self.title_label = tk.Label( master = self.title_frame, text = 'Solver Data Farming', width = 50,
                                    font = 'Calibir 15 bold')
        self.title_label.grid( row = 0, column = 0)
        
        # Option menu to select solver
        self.solver_select_label = tk.Label( master = self.solver_select_frame, text = 'Select Solver:', width = 20,
                                    font = 'Calibir 13')
        self.solver_select_label.grid( row = 0, column = 0)
        
        # Variable to store selected solver
        self.solver_var = tk.StringVar()
        
        #Directory of solver names
        self.solver_list = solver_unabbreviated_directory
        
        self.solver_select_menu = ttk.OptionMenu(self.solver_select_frame, self.solver_var, 'Solver', *self.solver_list, command = self.show_solver_factors)
        self.solver_select_menu.grid(row = 0, column = 1)
        
        # Load design selection
        self.load_design_label =  tk.Label( master = self.solver_select_frame, text = ' OR ', width = 20,
                                    font = 'Calibir 13')
        self.load_design_label.grid( row = 0, column = 2)
        self.load_design_button = tk.Button(master = self.solver_select_frame, text = 'Load Solver Design', font = 'Calibir 11',
                                            width = 20, command = self.load_solver_design)
        self.load_design_button.grid( row = 0, column = 3)
        
        #Dictonaries to hold experiment info
        self.experiment_pairs = {} #Dictionary to hold all experiment/ problem pairs, contains list of experiment, problem name, problem factors, then model factors
        self.select_pair_vars ={}
        self.macro_rep_vars = {}
        self.post_rep_vars= {}
        self.norm_rep_vars = {}
        self.run_buttons = {}
        self.post_buttons = {}
        self.norm_buttons = {}
        self.save_buttons = {}
        
        
        
    def clear_frame(self, frame):
        for widget in frame.winfo_children():
            widget.destroy()
    
    def load_solver_design(self):
    
        #Clear previous selections
        self.clear_frame(frame = self.solver_frame) #Clear previous solver selections
        self.clear_frame(frame = self.design_frame) # Clear design selections
        self.clear_frame(frame = self.problem_model_factors_frame) # Clear problem and model factor selections
        self.clear_frame(frame = self.problem_select_frame) # Clear problem selection widgets
        self.clear_frame(frame = self.design_view_frame) # Clear design tree
        self.clear_frame(self.create_pair_frame)
            
        
        # Initialize frames
        self.solver_frame.grid(row = 2, column = 0)
        self.solver_frame.grid_rowconfigure(0, weight =1)
        self.solver_frame.grid_columnconfigure(0, weight =1)
        self.solver_frame.grid_columnconfigure(1, weight =1)
        self.solver_frame.grid_columnconfigure(2, weight =1)
        self.solver_frame.grid_columnconfigure(3, weight =1)
        
        self.loaded_design = True # Design was loaded by user

        # Create column for model factor names
        self.headername_label = tk.Label(master = self.solver_frame, text = 'Default Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)
        
        # Create column for factor type
        self.headertype_label = tk.Label(master = self.solver_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
       
        
        # List to hold default values
        self.default_values_list = []
        self.fixed_str = {}
        
        # Create column for factor default values
        self.headerdefault_label = tk.Label(master = self.solver_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
            
        # Name of design csv file
        self.csv_filename = filedialog.askopenfilename()
        
        # convert loaded design to data frame
        self.design_table = pd.read_csv(self.csv_filename, index_col=False)
        
        # Get design information from table
        self.solver_name = self.design_table.at[1, 'Solver Name']
        self.design_type = self.design_table.at[1, 'Design Type']
        self.n_stacks = self.design_table.at[1, 'Number Stacks']
        
        
        # determine what factors are included in design
        self.factor_status = {} #dictionary that contains true/false for wheither factor is in design
        for col in self.design_table.columns[1:-3]: # col correspond to factor names, exclude index and information cols
            factor_set = set(self.design_table[col])
            
            if len(factor_set) > 1:
                design_factor = True
            else:
                design_factor = False
                
            self.factor_status[col] = design_factor
            
            
            
    
            
        # get default values for fixed factors
        self.default_factors = {} #contains only factors not in design, factor default vals input as str
        for factor in self.factor_status:
            if self.factor_status[factor] == False:
                self.default_factors[factor] = self.design_table.at[1, factor]
                    
  
        
        self.solver_class = solver_directory[self.solver_name]
        self.solver_object = self.solver_class()

        #Display model name
        self.solver_name_label = tk.Label( master = self.solver_select_frame, text = 'Selected Solver: ' + self.solver_name, font = "Calibri 13", width = 40  )
        self.solver_name_label.grid( row = 0, column = 4, sticky = tk.W)
        
        # Allow user to change default values
        factor_idx = 0
        self.factor_que_length = 1
        for  factor in self.default_factors:
            
            self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
            self.factor_description = self.solver_object.specifications[factor].get("description")
            self.factor_default = self.default_factors[factor]
            
            
            self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
            
            # Convert datatype to string for display
            if self.factor_datatype == int:
                self.str_type = 'int'
            elif self.factor_datatype == float:
                self.str_type = 'float'
            elif self.factor_datatype == bool:
                self.str_type = 'bool'
       
            # Add label for factor names
            self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
            self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
            
            # Add label for factor type
            self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
            self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
            
            #Add entry box for default value for int and float
            self.default_value= tk.StringVar()
            if self.factor_datatype == int or self.factor_datatype == float:
                self.default_entry = tk.Entry( master = self.solver_frame, width = 10, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.default_values_list.append(self.default_value)
                
            # Add option menu for bool factors
            elif self.factor_datatype == bool:
                self.default_value.set(self.factor_default) #Set default bool option
                self.bool_menu = ttk.OptionMenu(self.solver_frame, self.default_value, self.factor_default, 'TRUE', 'FALSE')
                self.bool_menu.grid( row = self.factor_que_length, column = 2, sticky = tk.N + tk.W)
                self.default_values_list.append(self.default_value)
 
            self.factor_que_length += 1
            factor_idx += 1
     
                    
        self.show_design_options() # run function to design creation options
        self.display_design_tree()


    def show_solver_factors(self, *args):
        
        
        #Initalize frames
        self.solver_frame.grid_rowconfigure(0, weight = 0)
        self.solver_frame.grid_columnconfigure(0, weight = 1)
        self.solver_frame.grid_columnconfigure(1, weight = 1)
        self.solver_frame.grid_columnconfigure(2, weight = 1)
        self.solver_frame.grid_columnconfigure(3, weight = 1)
        self.solver_frame.grid_columnconfigure(4, weight = 1)
        self.solver_frame.grid_columnconfigure(5, weight = 1)
        self.solver_frame.grid_columnconfigure(6, weight = 1)
        
       
        # Clear previous selections    
        self.clear_frame(frame = self.solver_frame) #Clear previous solver selections
        self.clear_frame(frame = self.design_frame) # Clear design selections
        self.clear_frame(frame = self.problem_model_factors_frame) # Clear problem and model factor selections
        self.clear_frame(frame = self.problem_select_frame) # Clear problem selection widgets
        self.clear_frame(frame = self.design_view_frame) # Clear design tree
        self.clear_frame(self.create_pair_frame)
        
        self.loaded_design = False # design was not loaded by user
        
        # Create column for solver factor names
        self.headername_label = tk.Label(master = self.solver_frame, text = 'Solver Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)
        

        
        # Create column for factor type
        self.headertype_label = tk.Label(master = self.solver_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
        
        
        # Create column for factor default values
        self.headerdefault_label = tk.Label(master = self.solver_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
        
        # Create column for factor check box
        self.headercheck_label = tk.Label(master = self.solver_frame, text = 'Include in Experiment', font = "Calibri 13 bold", width = 20 )
        self.headercheck_label.grid(row = 0, column = 3, sticky = tk.N + tk.W)
        
        # Create header for experiment options
        self.headercheck_label = tk.Label(master = self.solver_frame, text = 'Experiment Options', font = "Calibri 13 bold", width = 50 )
        self.headercheck_label.grid(row = 0, column = 4, columnspan = 3)
        
        # Get solver info from dictionary
        self.selected_solver = self.solver_var.get()
        self.solver_object = self.solver_list[self.selected_solver]()
        self.solver_name = self.solver_object.name # name of solver used for save files
        
        
        entry_width = 10
        
        # Widget lists
        self.default_widgets = {}
        self.check_widgets = {}
        self.min_widgets = {}
        self.max_widgets = {}
        self.dec_widgets = {}
        self.description_buttons = {}
        
        
               
        # Initial variable values
        self.factor_que_length = 1
        self.default_values_list = []
        self.checkstate_list=[]
        self.min_list = []
        self.max_list = []
        self.dec_list = []
        # self.descriptions = {} #used for description pop ups
        
       
        for  factor in self.solver_object.specifications:
            
            self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
            factor_description = self.solver_object.specifications[factor].get("description")
            self.factor_default = self.solver_object.specifications[factor].get("default")
            
            # self.descriptions[factor] = factor_description
            
            
            # Add label for factor names
            display_name = f"{factor} - {factor_description}"
            self.factorname_label = tk.Label (master = self.solver_frame, text = display_name, font = "Calibri 13", width = 80, anchor = 'w')
            self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
            
            
            
            
            if self.factor_datatype == float:
            
                self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'float'
    
                # # Add label for factor names
                # self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
               
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.solver_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                
                # Add check box
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.solver_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                # Add entry box for min val
                self.min_frame = tk.Frame (master = self.solver_frame)
                self.min_frame.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W )
                
                self.min_label = tk.Label(master = self.min_frame, text = 'Min Value', font = "Calibri 13", width = 10 )
                self.min_label.grid( row = 0, column = 0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry( master = self.min_frame, width = 10, textvariable = self.min_val, justify = 'right')
                self.min_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.min_list.append(self.min_val)    
                
                self.min_widgets[factor] = self.min_entry
                
                self.min_entry.configure(state = 'disabled')
                
                # Add entry box for max val
                self.max_frame = tk.Frame (master = self.solver_frame)
                self.max_frame.grid( row = self.factor_que_length, column = 5, sticky = tk.N + tk.W )
                
                self.max_label = tk.Label(master = self.max_frame, text = 'Max Value', font = "Calibri 13", width = 10 )
                self.max_label.grid( row = 0, column = 0) 
                
                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry( master = self.max_frame, width = 10, textvariable = self.max_val, justify = 'right')
                self.max_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
               
                self.max_list.append(self.max_val)    
                
                self.max_widgets[factor] = self.max_entry
                
                self.max_entry.configure(state = 'disabled')
                
                # Add entry box for editable decimals
                self.dec_frame = tk.Frame (master = self.solver_frame)
                self.dec_frame.grid( row = self.factor_que_length, column = 6, sticky = tk.N + tk.W )
                
                self.dec_label = tk.Label(master = self.dec_frame, text = '# Decimals', font = "Calibri 13", width = 10 )
                self.dec_label.grid( row = 0, column = 0) 
                
                self.dec_val = tk.StringVar()
                self.dec_entry = tk.Entry( master = self.dec_frame, width = 10, textvariable = self.dec_val, justify = 'right')
                self.dec_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.dec_list.append(self.dec_val)  
                
                self.dec_widgets[factor] = self.dec_entry
                
                self.dec_entry.configure(state = 'disabled')
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                # # Add label for factor names
                # self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.solver_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.solver_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                # Add entry box for min val
                self.min_frame = tk.Frame (master = self.solver_frame)
                self.min_frame.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W )
                
                self.min_label = tk.Label(master = self.min_frame, text = 'Min Value', font = "Calibri 13", width = 10 )
                self.min_label.grid( row = 0, column = 0)
                self.min_val = tk.StringVar()
                self.min_entry = tk.Entry( master = self.min_frame, width = 10, textvariable = self.min_val, justify = 'right')
                self.min_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
                
                self.min_list.append(self.min_val)    
                
                self.min_widgets[factor] = self.min_entry
                
                self.min_entry.configure(state = 'disabled')
                
                # Add entry box for max val
                self.max_frame = tk.Frame (master = self.solver_frame)
                self.max_frame.grid( row = self.factor_que_length, column = 5, sticky = tk.N + tk.W )
                
                self.max_label = tk.Label(master = self.max_frame, text = 'Max Value', font = "Calibri 13", width = 10 )
                self.max_label.grid( row = 0, column = 0) 
                
                self.max_val = tk.StringVar()
                self.max_entry = tk.Entry( master = self.max_frame, width = 10, textvariable = self.max_val, justify = 'right')
                self.max_entry.grid( row = 0, column = 1, sticky = tk.N + tk.W)
               
                self.max_list.append(self.max_val)    
                
                self.max_widgets[factor] = self.max_entry
                
                self.max_entry.configure(state = 'disabled') 
                
                self.factor_que_length += 1
                
            elif self.factor_datatype == bool:
            
                self.solver_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'bool'
    
                # # Add label for factor names
                # self.factorname_label = tk.Label (master = self.solver_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                # self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.solver_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add option menu for true/false
                self.default_value = tk.StringVar() #Variable to store selected bool factor state
                self.default_value.set('TRUE') #Set default bool option
                self.bool_menu = ttk.OptionMenu(self.solver_frame, self.default_value, 'TRUE', 'TRUE', 'FALSE')
                self.bool_menu.grid( row = self.factor_que_length, column = 2, sticky = tk.N + tk.W)
                
                
                
                # Default value if not included in design
                self.default_values_list.append(self.default_value)
                
                    
                # Add checkbox 
                self.checkstate = tk.BooleanVar()
                self.checkbox = tk.Checkbutton( master = self.solver_frame, variable = self.checkstate,
                                               command = self.include_factor)
                self.checkbox.grid( row = self.factor_que_length, column = 3, sticky = 'nsew')
                self.checkstate_list.append(self.checkstate)
                
                self.check_widgets[factor] = self.checkbox
                
                self.factor_que_length += 1
                
        # # Button for factor description
        # self.description_button = tk.Button(master = self.solver_frame, text = 'Factor Details', font = 'Calibri 11', width = 15, command = lambda: self.show_description_window(solver = self.solver_object ))
        # self.description_button.grid(row = self.factor_que_length + 1, column = 0)
        self.show_design_options() # run function to show design creation options
        # # self.bind_factor_descriptions()
    
    # def bind_factor_descriptions(self):
    #     for factor in self.description_buttons:
    #         print('factor', factor)
    #         self.description_buttons[factor].configure(command = lambda: self.show_description(factor))
            
        
        
        
    def show_design_options(self):
        # Design type selection menu
        self.design_frame.grid(row = 3, column = 0)
        self.design_frame.grid_rowconfigure(0, weight = 0)
        self.design_frame.grid_columnconfigure(0, weight = 1)
        self.design_frame.grid_columnconfigure(1, weight = 1)
        self.design_frame.grid_columnconfigure(2, weight = 1)
        self.design_frame.grid_columnconfigure(3, weight = 1)
        self.design_frame.grid_columnconfigure(4, weight = 1)
        
        # Input options from loaded designs
        if self.loaded_design == True:
            stack_display = self.n_stacks # same num of stacks as original loaded design
            design_display = self.design_type
        else:
            stack_display = 'Stacks'
            design_display = 'Design Type'
        
        self.design_type_label = tk.Label (master = self.design_frame, text = 'Select Design Type:', font = "Calibri 13", width = 20)
        self.design_type_label.grid( row = 0, column = 0)
        
        self.design_types_list = ['nolhs']
        self.design_var = tk.StringVar()
        self.design_type_menu = ttk.OptionMenu(self.design_frame, self.design_var, design_display, *self.design_types_list, command = self.enable_stacks)
        self.design_type_menu.grid(row = 0, column = 1)
        if self.loaded_design == True:
            self.design_type_menu.configure(state = 'disabled')
        
        #Stack selection menu
        self.stack_label = tk.Label (master = self.design_frame, text = "Select Number of Stacks:", font = "Calibri 13", width = 20)
        self.stack_label.grid( row =1, column = 0)
        self.stack_var = tk.StringVar()
        self.stack_var.set('1')
        self.stack_menu = ttk.Entry(master = self.design_frame,  width = 10, textvariable = self.stack_var, justify = 'right')
        self.stack_menu.grid( row = 1, column = 1)
        self.stack_menu.configure(state = 'disabled')
        
        # Name of design file entry
        self.design_filename_label = tk.Label (master = self.design_frame, text = "Name of design:", font = "Calibri 13", width = 20)
        self.design_filename_label.grid( row = 0, column = 2)
        self.design_filename_var = tk.StringVar() # variable to hold user specification of desing file name
        #timestamp = time.strftime("%Y%m%d%H%M%S")
        self.design_filename_var.set(self.solver_name)
        self.design_filename_entry = tk.Entry(master = self.design_frame,  width = 40, textvariable = self.design_filename_var, justify = 'right' )
        self.design_filename_entry.grid( row = 0, column = 3)
        
         
        # Create design button 
        if self.loaded_design == False:
            self.create_design_button = tk.Button(master = self.design_frame, text = 'Create Design', font = "Calibri 13", command = self.create_design , width = 20)
            self.create_design_button.grid( row = 0, column = 4)
            
        # Modify design button for loaded designs
        if self.loaded_design == True:
            
            self.mod_design_button = tk.Button(master = self.design_frame, text = 'Modify Design', font = "Calibri 13", command = self.mod_design , width = 20)
            self.mod_design_button.grid( row = 0, column = 4)
            
            self.continue_design_button = tk.Button(master = self.design_frame, text = 'Continue w/o Changes', font = "Calibri 13", command = self.con_design , width = 25)
            self.continue_design_button.grid( row = 1, column = 4)
            
    def mod_design(self):
        
        self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
        factor_index = 0
        for factor in self.default_factors:
            #self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
            new_val = self.default_values[factor_index]
            self.design_table[factor] = new_val
            factor_index += 1
            
        self.design_filename = self.design_filename_var.get() # name of design file specified by user
        
        self.csv_filename = f'./data_farming_experiments/{self.design_filename}_design.csv'
        
        self.design_table.to_csv(self.csv_filename, index = False)

        self.experiment = DataFarmingMetaExperiment(csv_filename = self.csv_filename)
        self.select_problem()
        self.display_design_tree()
        
                                                    
        
        
        
    def con_design(self):
        self.experiment = DataFarmingMetaExperiment(csv_filename = self.csv_filename)
        self.select_problem()
                                                       
        
    
    
    def select_problem(self):
       
        # Problem selection frame
        self.problem_select_frame.grid( row = 5, column = 0)
        self.problem_select_frame.grid_rowconfigure(0, weight = 1)
        self.problem_select_frame.grid_columnconfigure(0, weight = 1)
        self.problem_select_frame.grid_columnconfigure(1, weight = 1)
        self.problem_select_frame.grid_columnconfigure(2, weight = 1)
        
        
        self.update_problem_list_compatability() #Check compatibility of solver problems, returns self.problem_list for option menu
        
        # Option menu to select problem
        self.problem_select_label = tk.Label( master = self.problem_select_frame, text = 'Select Problem:', width = 20,
                                    font = 'Calibir 13')
        self.problem_select_label.grid( row = 0, column = 0)
        
        self.problem_var = tk.StringVar() # Variable to store selected problem
          
        
        self.problem_select_menu = ttk.OptionMenu(self.problem_select_frame, self.problem_var, 'Problem', *self.problem_list, command = self.show_problem_factors)
        self.problem_select_menu.grid(row = 0, column = 1)   
        
        # Display model name
        self.model_name = ""
        self.model_label = tk.Label(master = self.problem_select_frame, text = 'Model: ' + self.model_name, font = "Calibri 13")
        self.model_label.grid(row = 0, column = 2)
        
      
    def enable_stacks(self, *args):
        if self.loaded_design == False: #cannot change stacks for loaded design
            self.stack_menu.configure(state = 'normal')
        
    def enable_create_design_button(self, *args):
        self.create_design_button.configure(state = 'normal')
        
        

    
    
    def create_design(self, *args):

        
        #Export design factors

        
        self.n_stacks = self.stack_var.get() # user input for num stacks
        self.design_type = self.design_var.get() #user input for design type
        
        # self.problem_fixed_factors = {} # holds fixed factors for problem to be used in design
        # self.model_fixed_factors ={} # holds fixed factors of model to be used in design
        self.fixed_factors = {} # holds fixed factors of solver to be used in design
        self.factor_index = 0
         
        
        # List to hold names of all factors part of model to be displayed in csv
        self.factor_names = [] # names of solver factors included in experiment
        self.def_factor_names = [] # names of default and cross design solver factors
        self.problem_factor_names = [] # names of all problem factors
        self.model_factor_names = [] # names of all model factors
        
 
      
        # Write solver factors file
        
        self.design_filename = self.design_filename_var.get() # name of design file specified by user
        
        with open(f"./data_farming_experiments/{self.design_filename}.txt", "w") as self.solver_design_factors:
            self.solver_design_factors.write("")
            
        
       
            
        if self.loaded_design == False: # get factor settings for created design (not loaded)
        
            # List used for parsing design file
            self.factor_headers = [] 
           
            # Lists that hold factor information set by user
            self.check_values = [self.checkstate.get() for self.checkstate in self.checkstate_list] # checkstate of each factor
            self.default_values = [self.default_value.get() for self.default_value in self.default_values_list] # default value of each factor
            self.min_values = [self.min_val.get() for self.min_val in self.min_list] # max value of each int & float factor
            self.max_values = [self.max_val.get() for self.max_val in self.max_list] # min value of each int & float factor
            self.dec_values = [self.dec_val.get() for self.dec_val in self.dec_list] # dec value of each float factor
            
            
            # values to index through factors
            self.factor_index = 0
            self.maxmin_index = 0
            self.dec_index = 0
            
            #Dictionary used for tree view display of fixed factors
            self.solver_fixed_str = {}
            self.def_factors = [] # list of factors not included in design
            
           
            # Get solver experiment information    
            for  factor in self.solver_object.specifications:
                
                self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
               
                self.factor_default = self.default_values[self.factor_index]
                self.factor_include = self.check_values[self.factor_index]
                
                
               
               
                if self.factor_include == True:
                    
                    # # Add factor to list of design factors in order that will be varied
                    # self.factor_headers.append(factor)
                    # Factor names in csv are unedited
                    # self.factor_names.append(factor)
                
                    if self.factor_datatype == float or self.factor_datatype == int:
                        # Add factor to list of design factors in order that will be varied
                        self.factor_headers.append(factor)
                        # Factor names in csv are unedited
                        self.factor_names.append(factor)
                        
                        self.factor_min = str(self.min_values[self.maxmin_index])
                        self.factor_max = str(self.max_values[self.maxmin_index])
                        self.maxmin_index += 1
                        
                        if self.factor_datatype == float:
                            self.factor_dec = str(self.dec_values[self.dec_index])
                            self.dec_index += 1
                            
                        elif self.factor_datatype == int:
                            self.factor_dec = '0'
                            
                        self.data_insert = self.factor_min + ' ' + self.factor_max + ' ' + self.factor_dec
                        with open(f"./data_farming_experiments/{self.design_filename}.txt", "a") as self.solver_design_factors:
                            self.solver_design_factors.write(self.data_insert + '\n')  
                            
                    elif self.factor_datatype == bool: 
                            
                            factor_options = [True,False] # list of values factor can take, temp hard coded for true/false
                            self.cross_design_factors[factor] = factor_options #add factor to cross design dictionary
                            display = ""
                            for opt in factor_options:
                                opt_str = str(opt)
                                display += opt_str + '/'
                            self.solver_fixed_str[factor] = display[:-1] # add cross design factor to fixed str for design table
                            self.def_factor_names.append(factor + ' (cross)') # list of factor names in cross design
                       
                
                # Include fixed default values in design
                if self.factor_include == False:
                    
                    # Factor names in csv have "(default)" appended to end
                    self.def_factors.append(factor)
                    self.def_factor_names.append(factor + ' (fixed)')
                    
                    # Values to be placed in tree view of design
                    self.solver_fixed_str[factor] = self.factor_default
                   
                    if self.factor_datatype == float or self.factor_datatype == int:
                        self.maxmin_index += 1
                   
                    # Add default values to exeriment and set to correct datatype
                    if self.factor_datatype == float:
                        self.fixed_factors[factor] = float(self.factor_default)
                        self.dec_index += 1
                        
                    elif self.factor_datatype == int:
                        self.fixed_factors[factor] = int(self.factor_default)
                        
                # bool values currently not able to be included in design
                if self.factor_datatype == bool:
                    if self.factor_default == 'TRUE':
                        bool_val = bool(1)
                    else:
                        bool_val = bool(0) 
                    self.fixed_factors[factor] = bool_val
                    
       
                self.factor_index += 1
                
            # Create design csv file with headers
            self.all_factor_headers = self.factor_names +  self.def_factor_names # combine factor names
                
            # Create solver factor design from .txt file of factor settings.
            # Hard-coded for NOLHS.
            self.design_filename = self.design_filename_var.get() # name of design file specified by user
            
            #self.filename = 'solver_factors' # base for all design file names, temp, turn into ask dialog
            
            if self.loaded_design == False:
                self.csv_filename = f'./data_farming_experiments/{self.design_filename}_design.csv' # used to display design tree
            
            self.experiment = DataFarmingMetaExperiment(solver_name = self.solver_name,
                                                           solver_factor_headers = self.factor_names,
                                                           n_stacks = self.n_stacks,
                                                           design_type = self.design_type,
                                                           solver_factor_settings_filename = self.design_filename,
                                                           design_filename= None,
                                                           solver_fixed_factors = self.fixed_factors,
                                                           cross_design_factors = self.cross_design_factors,
                                                           csv_filename = None
                                                           )
            
           
                       
 
        self.display_design_tree() # show created design
        self.select_problem() # show problem selection menu 
       
      
        
    # Used to display the design tree for both created and loaded designs
    def display_design_tree(self):
      
        #Initialize design tree
        self.design_view_frame.grid( row = 4, column = 0)
        self.design_view_frame.grid_rowconfigure( 0, weight = 0)
        self.design_view_frame.grid_rowconfigure( 1, weight = 1)
        self.design_view_frame.grid_columnconfigure( 0, weight = 1)
        self.design_view_frame.grid_columnconfigure( 1, weight = 1)
        
      
   
        self.design_tree = ttk.Treeview( master = self.design_view_frame)
        self.design_tree.grid(row = 1, column = 0, sticky = 'nsew')
      
       
        
        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=('Calibri', 13, 'bold'))
        self.style.configure("Treeview", foreground="black", font = ('Calibri', 13))
        self.design_tree.heading( '#0', text = 'Design #' )
        
        # Get design point values from csv
        design_table = pd.read_csv(self.csv_filename, index_col="Design #")
        num_dp = len(design_table) #used for label
        self.create_design_label = tk.Label( master = self.design_view_frame, text = f'Total Design Points: {num_dp}', font = "Calibri 13 bold", width = 50)
        self.create_design_label.grid(row = 0, column = 0, sticky = tk.W)
        
        # Enter design values into treeview
        self.design_tree['columns'] = tuple(design_table.columns)[:-3]
       
        
        
        
        for column in design_table.columns[:-3]:
            self.design_tree.heading( column, text = column)
            self.design_tree.column(column, width = 100)
            
        for  index, row in design_table.iterrows():
            print('row', row)
            
            self.design_tree.insert("", index, text = index, values = tuple(row)[:-3])
        
  
       
      
            # Create a horizontal scrollbar
        xscrollbar = ttk.Scrollbar(master = self.design_view_frame, orient="horizontal", command= self.design_tree.xview)
        xscrollbar.grid(row = 2, column = 0, sticky = 'nsew')
        
        # Configure the Treeview to use the horizontal scrollbar
        self.design_tree.configure(xscrollcommand=xscrollbar.set) 
    def show_problem_factors(self, *args):
        
        # self.show_experiment_options() # show options for experiment creation
        
        #Initalize frames
        self.problem_model_factors_frame.grid(row = 6, column = 0)
        self.problem_model_factors_frame.grid_rowconfigure(0, weight = 1)  
        self.problem_model_factors_frame.grid_columnconfigure(0, weight = 1)
        self.problem_model_factors_frame.grid_columnconfigure(1, weight = 1)
        self.problem_model_factors_frame.grid_columnconfigure(2, weight = 1)
        
        self.create_pair_frame.grid( row = 7, column = 0)
        self.create_pair_frame.grid_rowconfigure(0, weight = 1)
        self.create_pair_frame.grid_columnconfigure(0, weight = 1)
        self.create_pair_frame.grid_columnconfigure(1, weight = 1)
        self.create_pair_frame.grid_columnconfigure(2, weight = 1)
       
        
        #Clear previous selections
        self.clear_frame(self.problem_model_factors_frame) # clear problem-model factor selections
        self.clear_frame(self.create_pair_frame) # clear create pair buttons
        self.model_label.destroy() #remove old model label
        
            
        self.problem_frame = tk.Frame(master = self.problem_model_factors_frame)
        self.model_frame = tk.Frame(master = self.problem_model_factors_frame)
         
         
        self.problem_frame.grid(row = 1, column = 0)
        self.problem_frame.grid_rowconfigure(0, weight = 0)
        self.problem_frame.grid_columnconfigure(0, weight = 1)
        self.problem_frame.grid_columnconfigure(1, weight = 1)
        self.problem_frame.grid_columnconfigure(2, weight = 1)
        self.problem_frame.grid_columnconfigure(3, weight = 1)
        self.problem_frame.grid_columnconfigure(4, weight = 1)
        self.problem_frame.grid_columnconfigure(5, weight = 1)
        self.problem_frame.grid_columnconfigure(6, weight = 1)
         
        self.model_frame.grid(row = 0, column = 1 )
        self.model_frame.grid_rowconfigure(0, weight = 0)
        self.model_frame.grid_columnconfigure(0, weight = 1)
        self.model_frame.grid_columnconfigure(1, weight = 1)
        self.model_frame.grid_columnconfigure(2, weight = 1)
        
        
        


        # Create column for problem factor names
        self.headername_label = tk.Label(master = self.problem_model_factors_frame, text = 'Problem Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 0, sticky = tk.N + tk.W)
        
        # Create column for problem factor type
        self.headertype_label = tk.Label(master = self.problem_model_factors_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 15, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 1, sticky = tk.N + tk.W)
        
        
        # Create column for problem factor default values
        self.headerdefault_label = tk.Label(master = self.problem_model_factors_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 2, sticky = tk.N + tk.W)
        
        # Create column for model factor names
        self.headername_label = tk.Label(master = self.problem_model_factors_frame, text = 'Model Factors', font = "Calibri 13 bold", width = 20, anchor = 'w')
        self.headername_label.grid(row = 0, column = 3, sticky = tk.N + tk.W)
        
        # Create column for model factor type
        self.headertype_label = tk.Label(master = self.problem_model_factors_frame, text = 'Factor Type', font = "Calibri 13 bold", width = 20, anchor = 'w' )
        self.headertype_label.grid(row = 0, column = 4, sticky = tk.N + tk.W)
        
        
        # Create column for model factor default values
        self.headerdefault_label = tk.Label(master = self.problem_model_factors_frame, text = 'Default Value', font = "Calibri 13 bold", width = 20 )
        self.headerdefault_label.grid(row = 0, column = 5, sticky = tk.N + tk.W)
        
      
        entry_width = 10
        
        # Widget lists
        self.default_widgets = {}
        
        # Initial variable values
        self.factor_que_length = 1
        self.problem_default_values_list = []
        self.model_default_values_list = []
        
        # Get problem info from dictionary
        self.selected_problem = self.problem_var.get()
        self.problem_object = problem_unabbreviated_directory[self.selected_problem]()
        self.problem_name = self.problem_object.name # name of problem used for save files
   
        # Get model info from dictonary
        self.model_problem_dict = model_problem_class_directory # directory that relates problem name to model class
        self.model_object = self.model_problem_dict[self.selected_problem]()
        self.model_name = self.model_object.name # name of model that relates to problem 
        
              
        # Display model name
        self.model_label = tk.Label(master = self.problem_select_frame, text = 'Model: ' + self.model_name, font = "Calibri 13")
        self.model_label.grid(row = 0, column = 2)
        
        
        
        for  factor in self.problem_object.specifications:
            
            self.factor_datatype = self.problem_object.specifications[factor].get("datatype")
            self.factor_description = self.problem_object.specifications[factor].get("description")
            self.factor_default = self.problem_object.specifications[factor].get("default")
            
            self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
        
            
            
            if self.factor_datatype == float:
            
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'float'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
               
                
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.problem_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
               
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.problem_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == list:
                
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'list'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                #Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.problem_default_values_list.append(self.default_value)
                
            
                self.factor_que_length += 1
                
            elif self.factor_datatype == tuple:
                
                self.problem_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'tuple'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 0, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 1, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =2, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.problem_default_values_list.append(self.default_value)
                
                self.factor_que_length += 1
                
        self.factor_que_length = 1
                
        for  factor in self.model_object.specifications:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
            self.factor_description = self.model_object.specifications[factor].get("description")
            self.factor_default = self.model_object.specifications[factor].get("default")
            
            
            #Values to help with formatting
            entry_width = 10
            
            self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
        
            
            
            if self.factor_datatype == float:
                
                self.str_type = 'float'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)

                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.model_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
          
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == int:
            
                self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'int'
    
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W)
                #Display original default value
                self.default_entry.insert(0, self.factor_default)
                self.model_default_values_list.append(self.default_value)
                
                self.default_widgets[factor] = self.default_entry
                
                
                self.factor_que_length += 1
            
            elif self.factor_datatype == list:
                
                self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'list'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                #Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.model_default_values_list.append(self.default_value)
                
            
                self.factor_que_length += 1
                
            elif self.factor_datatype == tuple:
                
                self.model_frame.grid_rowconfigure(self.factor_que_length, weight =1)
                
                self.str_type = 'tuple'
               
                # Add label for factor names
                self.factorname_label = tk.Label (master = self.problem_model_factors_frame, text = factor, font = "Calibri 13", width = 30, anchor = 'w')
                self.factorname_label.grid( row = self.factor_que_length, column = 3, sticky = tk.N + tk.W)
                
                # Add label for factor type
                self.factortype_label = tk.Label (master = self.problem_model_factors_frame, text = self.str_type, font = "Calibri 13", width = 20, anchor = 'w')
                self.factortype_label.grid( row = self.factor_que_length, column = 4, sticky = tk.N + tk.W)
                
                # Add entry box for default value
                default_len = len(str(self.factor_default))
                if default_len > entry_width:
                    entry_width = default_len
                    if default_len > 150:
                        entry_width = 150
                self.default_value= tk.StringVar()
                self.default_entry = tk.Entry( master = self.problem_model_factors_frame, width = entry_width, textvariable = self.default_value, justify = 'right')
                self.default_entry.grid( row =self.factor_que_length, column =5, sticky = tk.N + tk.W, columnspan = 5)
                #Display original default value
                self.default_entry.insert(0, str(self.factor_default))
                self.model_default_values_list.append(self.default_value)
                
               
                self.factor_que_length += 1  
                
        # Save pair name
        self.pair_name_label = tk.Label(master = self.create_pair_frame, text = 'Save pair as:' , width = 20, font = 'Calibri 13')
        self.pair_name_label.grid(row = 0, column = 0)
        self.pair_name_var = tk.StringVar() #store name of design-problem pair
        self.pair_name_var.set(self.solver_name + '_' + self.problem_name)
        self.pair_name_entry = tk.Entry( master = self.create_pair_frame, width = 20, textvariable = self.pair_name_var, justify = 'right')
        self.pair_name_entry.grid(row = 0, column = 1)
        
        # Create design-problem pair button
        self.add_pair_button = tk.Button(master = self.create_pair_frame, text = 'Add Design-Problem Pair', font = 'Calibri 13',
                                         width = 20, command = self.add_pair)
        self.add_pair_button.grid(row = 0, column = 2)
        
       
        
        
        
                
    def add_pair(self):
        
        # Experiment frames
        self.experiment_frame.grid( row = 8, column = 0)
        self.experiment_frame.grid_rowconfigure(0, weight = 1)
        self.experiment_frame.grid_columnconfigure(0, weight = 1)
        self.experiment_frame.grid_columnconfigure(1, weight = 1)
        self.experiment_frame.grid_columnconfigure(2, weight = 1)
        self.experiment_frame.grid_columnconfigure(3, weight = 1)
        self.experiment_frame.grid_columnconfigure(4, weight = 1)
        self.experiment_frame.grid_columnconfigure(5, weight = 1)
        self.experiment_frame.grid_columnconfigure(6, weight = 1)
        self.experiment_frame.grid_columnconfigure(7, weight = 1)
        self.experiment_frame.grid_columnconfigure(8, weight = 1)
        self.experiment_frame.grid_columnconfigure(9, weight = 1)
        self.experiment_frame.grid_columnconfigure(10, weight = 1)
        self.experiment_frame.grid_columnconfigure(11, weight = 1)
        self.experiment_frame.grid_columnconfigure(12, weight = 1)
            
        
        if len(self.experiment_pairs) == 0: # for first added pair create header
            
            self.pair_name_header = tk.Label( master = self.experiment_frame, text = 'Generated Pairs', width = 20, font = 'Calibri 13 bold')
            self.pair_name_header.grid(row = 0, column = 1)
            
            self.mac_rep_label = tk.Label(master = self.experiment_frame, text = '# Macro Replications', font = "Calibri 13 bold", width = 20)
            self.mac_rep_label.grid(row = 0, column = 2)
            
            self.post_rep_label = tk.Label(master = self.experiment_frame, text = '# Post Replications', font = "Calibri 13 bold", width = 20)
            self.post_rep_label.grid(row = 0, column = 4)
            
            self.crn_budget_label = tk.Label(master = self.experiment_frame, text = 'CRN Across Budget?', font = "Calibri 13 bold", width = 20)
            self.crn_budget_label.grid(row = 0, column = 5)
            
            self.crn_macro_label = tk.Label(master = self.experiment_frame, text = 'CRN Across Macro?', font = "Calibri 13 bold", width = 20)
            self.crn_macro_label.grid(row = 0, column = 6)
            
            self.norm_rep_label = tk.Label(master = self.experiment_frame, text = '# Normalization Replications', font = "Calibri 13 bold", width = 0)
            self.norm_rep_label.grid(row = 0, column = 8)
            
            self.crn_macro_label = tk.Label(master = self.experiment_frame, text = 'CRN Across Initial Option?', font = "Calibri 13 bold", width = 25)
            self.crn_macro_label.grid(row = 0, column = 9)
            

        self.problem_default_values = [self.default_value.get() for self.default_value in self.problem_default_values_list]
        self.model_default_values = [self.default_value.get() for self.default_value in self.model_default_values_list]
        
        self.problem_fixed_factors = {} # holds fixed factors for problem to be used in design
        self.model_fixed_factors ={} # holds fixed factors of model to be used in design
        self.factor_index = 0
         
        
      
        
        # Get problem default factor information    
        for  factor in self.problem_object.specifications:
            
            self.factor_datatype = self.problem_object.specifications[factor].get("datatype")
      
            self.factor_default = self.problem_default_values[self.factor_index]
            print("default", self.factor_default)
            # self.problem_factor_names.append( factor + ' (fixed)')
            
            # # Values to be placed in tree view of design
            # self.problem_fixed_str[factor] = self.factor_default
            
          
           
            # Add default values to exeriment and set to correct datatype
            if self.factor_datatype == float:
                self.problem_fixed_factors[factor] = float(self.factor_default)
                
            if self.factor_datatype == int:
                self.problem_fixed_factors[factor] = int(self.factor_default)
                
            if self.factor_datatype == list:
                self.problem_fixed_factors[factor] = ast.literal_eval(self.factor_default)
                
            if self.factor_datatype == tuple:
                last_val = self.factor_default[-2]
                tuple_str = self.factor_default[1:-1].split(",")
                print('tuple string', )
                # determine if last tuple value is empty
                if last_val != ",":
                    self.problem_fixed_factors[factor] = tuple(float(s) for s in tuple_str)
                else: 
                    tuple_exclude_last = tuple_str[:-1]
                    float_tuple = [float(s) for s in tuple_exclude_last]
                    self.problem_fixed_factors[factor] = tuple(float_tuple)
                    print( 'final tuple', tuple(float_tuple) )
                    
   
            self.factor_index += 1
        
        # Get model default factor information
        self.factor_index = 0
        for  factor in self.model_object.specifications:
            
            self.factor_datatype = self.model_object.specifications[factor].get("datatype")
      
            self.factor_default = self.model_default_values[self.factor_index]
            
            # self.model_factor_names.append( factor + ' (fixed)')
            
            # # Values to be placed in tree view of design
            # self.model_fixed_str[factor] = self.factor_default
         
            # Add default values to exeriment and set to correct datatype
            if self.factor_datatype == float:
                self.model_fixed_factors[factor] = float(self.factor_default)
               
            if self.factor_datatype == int:
                self.model_fixed_factors[factor] = int(self.factor_default)
                
            if self.factor_datatype == list:
                self.model_fixed_factors[factor] = ast.literal_eval(self.factor_default)
            
            if self.factor_datatype == tuple:
                   
                tuple_str = tuple(self.factor_default[1:-1].split(","))
                self.model_fixed_factors[factor] = tuple(float(s) for s in tuple_str)
       
            self.factor_index += 1
        

        pair_name = self.pair_name_var.get() #name of problem pair
        self.experiment_pairs[pair_name] = [self.experiment, self.problem_object.name, self.problem_fixed_factors, self.model_fixed_factors] # Add experiment to dictionary by pair name
        
        self.show_experiment_options(pair_name = pair_name)


        
    def show_experiment_options(self, pair_name):
        # Show experiment options
        current_row = len(self.experiment_pairs)
        self.experiment_frame.grid_rowconfigure(current_row, weight = 1)

       
        
        # Pair name label
        self.pair_name_label = tk.Label(master = self.experiment_frame, text = pair_name, font = 'Calibri 11', width = 20)
        self.pair_name_label.grid( row = current_row, column = 1)
        
       # number macro replications
        self.mac_rep_var = tk.StringVar() # variable to store number of macro replication of experiment
        self.macro_rep_vars[pair_name] = self.mac_rep_var # store to macro list for all experiments
        
        self.mac_rep_entry = tk.Entry(master = self.experiment_frame, width = 10, textvariable = self.mac_rep_var, justify = 'right')
        self.mac_rep_entry.grid(row = current_row, column = 2)
        
        
        # Run experiment button
        self.run_exp_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Run', font = "Calibri 11", command = lambda: self.run_experiment(pair_name = pair_name))
        self.run_exp_button.grid( row = current_row, column = 3)
        self.run_buttons[pair_name] = self.run_exp_button # store to run button list for all experiments
        
        # Number post replications
        self.post_rep_var = tk.StringVar() # variable to store number of post replications of experiment
        self.post_rep_vars[pair_name] = self.post_rep_var # store to post rep list for all experiments
        
        self.post_rep_entry = tk.Entry(master = self.experiment_frame, width = 10, textvariable = self.post_rep_var, justify = 'right')
        self.post_rep_entry.grid(row = current_row, column = 4) 
        
        # CRN across budget for post rep
        crn_budget_var = tk.StringVar() # variable to hold true/false state of crn across budget
        self.crn_budget_menu = ttk.OptionMenu(self.experiment_frame, crn_budget_var, 'True', *['True', 'False'])
        self.crn_budget_menu.grid(row = current_row, column = 5)
        
        # CRN across macro for post rep
        crn_macro_var = tk.StringVar() # variable to hold true/false state of crn across macro reps
        self.crn_macro_menu = ttk.OptionMenu(self.experiment_frame, crn_macro_var, 'False', *['True', 'False'])
        self.crn_macro_menu.grid(row = current_row, column = 6)
        
        # Post process button experiment button
        self.post_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Post-Process', font = "Calibri 11", command = lambda: self.post_replicate(pair_name =pair_name, 
                                                                                                                                                                    crn_across_budget=crn_budget_var.get(),
                                                                                                                                                                    crn_across_macro=crn_macro_var.get()))
        self.post_button.grid( row = current_row, column = 7)
        self.post_buttons[pair_name] = self.post_button # store to post button list for all experiments
        self.post_button.configure( state = 'disabled')
        
        
        # Number normalization replications
       
        self.norm_rep_var = tk.StringVar() # variable to store number of normalization replications of experiment
        self.norm_rep_vars[pair_name] = self.norm_rep_var # store to norm rep list for all experiments
        
        self.norm_rep_entry = tk.Entry(master = self.experiment_frame, width = 10, textvariable = self.norm_rep_var, justify = 'right')
        self.norm_rep_entry.grid(row = current_row, column = 8) 
        
        # CRN across init option
        crn_init_var = tk.StringVar() # variable to hold true/false state of crn across initial option
        self.crn_init_menu = ttk.OptionMenu(self.experiment_frame, crn_init_var, 'True', *['True', 'False'])
        self.crn_init_menu.grid(row = current_row, column = 9)
        
        
        # Normalize experiment button
        self.norm_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Normalize', font = "Calibri 11", command = lambda: self.post_normalize(pair_name = pair_name,
                                                                                                                                                                 crn_across_init = crn_init_var.get()))
        self.norm_button.grid( row = current_row, column = 10)
        self.norm_buttons[pair_name] = self.norm_button # store to norm button list for all experiments
        self.norm_button.configure( state = 'disabled')
        
        # Save experiment button
        self.save_button = tk.Button( master = self.experiment_frame, width = 10, text = 'Save', font = "Calibri 11", command = lambda: self.save_results(pair_name = pair_name))
                                                                                                                                           
        self.save_button.grid( row = current_row, column = 11)
        self.save_buttons[pair_name] = self.save_button # store to save button list for all experiments
        self.save_button.configure( state = 'disabled')



    def run_experiment(self, pair_name):
        
        
        current_exp = self.experiment_pairs[pair_name][0] # experiment is first element in stored list
        n_macroreps = int(self.macro_rep_vars[pair_name].get())
        
        problem_name = self.experiment_pairs[pair_name][1]
        problem_fixed_factors = self.experiment_pairs[pair_name][2]
        model_fixed_factors = self.experiment_pairs[pair_name][3]


        # Run macroreplications at each design point.
        current_exp.run(n_macroreps=n_macroreps, 
                        problem_name=problem_name, 
                        problem_fixed_factors=problem_fixed_factors, 
                        model_fixed_factors=model_fixed_factors 
                        )

      
    
        self.run_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        self.post_buttons[pair_name].configure(state = 'normal')
        self.norm_buttons[pair_name].configure( state = 'disabled')
        self.save_buttons[pair_name].configure( state = 'disabled')
    
    
   
    def post_replicate(self, pair_name, crn_across_budget, crn_across_macro):
        
        
        current_exp = self.experiment_pairs[pair_name][0]
       
        # Specify the number of postreplications to take at each recommended solution
        # from each macroreplication at each design point.
        # self.postreps = self.post_rep_var.get() # number of post reps specified by user (string)
        n_postreps = int(self.post_rep_vars[pair_name].get())
        # n_postreps = int(self.postreps)
        
        # Specify the CRN control for postreplications.   
        if crn_across_budget == 'True':
            crn_across_budget = True 
        else:
            crn_across_budget = False
            
        if crn_across_macro == 'True':
            crn_across_macroreps = True 
        else:
            crn_across_macroreps = False 
        
        
        # Postprocess the experimental results from each design point.
        current_exp.post_replicate(n_postreps=n_postreps,
                                          crn_across_budget=crn_across_budget,
                                          crn_across_macroreps=crn_across_macroreps
                                          )
        
        self.post_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        self.norm_buttons[pair_name].configure(state = 'normal')
        self.save_buttons[pair_name].configure( state = 'disabled')
        
        
   
    def post_normalize(self, pair_name, crn_across_init):
 
        
        current_exp = self.experiment_pairs[pair_name][0]
        #Determine CRN
        if crn_across_init == 'True':
            crn_across_init_opt = True 
        else:
            crn_across_init_opt = False
        
        # Specify the number of postreplications to take at x0 and x*.
        n_postreps_init_opt = int(self.norm_rep_vars[pair_name].get())
        # self.normreps = self.norm_rep_var.get()
        # n_postreps_init_opt = int(self.normreps)
        
        current_exp.post_normalize(n_postreps_init_opt=n_postreps_init_opt,
                                          crn_across_init_opt=crn_across_init_opt
                                          )
        
        self.norm_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        self.save_buttons[pair_name].configure( state = 'normal')
        

    def save_results(self, pair_name):
        
        
        current_exp = self.experiment_pairs[pair_name][0]
        
        # Save experiment results file name
        export_csv_filename =  filedialog.asksaveasfilename(initialfile = pair_name + "_datafarming_experiment" )
        
        # Compute the performance metrics at each design point and print to csv.
        current_exp.report_statistics(solve_tols=[0.05, 0.10, 0.20, 0.50], csv_filename = export_csv_filename)
        
        self.save_buttons[pair_name].configure( bg = 'dark gray') # change color of button
        
        
    def update_problem_list_compatability(self):
        
        temp_problem_list = []
        temp_solver_name = self.solver_name
        
        for problem in problem_unabbreviated_directory:

            temp_problem = problem_unabbreviated_directory[problem] # problem object
            temp_problem_name = temp_problem().name

            temp_experiment = ProblemSolver(solver_name=temp_solver_name, problem_name=temp_problem_name)
            comp = temp_experiment.check_compatibility()

            if comp == "":
                temp_problem_list.append(problem)

        # from experiments.inputs.all_factors.py:
        self.problem_list = temp_problem_list # list of problems used for option menu 

           
    
    def include_factor(self, *args):

        self.check_values = [self.checkstate.get() for self.checkstate in self.checkstate_list]
        self.check_index = 0
        self.cat_index = 0
        self.cross_design_factors = {} # Dictionary to hold cross design factors and lists containing possible factor values
    
        # If checkbox to include in experiment checked, enable experiment option buttons
        for factor in self.solver_object.specifications:
                  
            # Get current checksate values from include experiment column
            self.current_checkstate = self.check_values[self.check_index]
            # Cross check factor type
            self.factor_datatype = self.solver_object.specifications[factor].get("datatype")
            
            # Disable / enable experiment option widgets depending on factor type
            if self.factor_datatype == float or self.factor_datatype == int:
                self.current_min_entry = self.min_widgets[factor]
                self.current_max_entry = self.max_widgets[factor]               
                
                             
                if self.current_checkstate == True:
                    self.current_min_entry.configure(state = 'normal')
                    self.current_max_entry.configure(state = 'normal')
                    
                elif self.current_checkstate == False:
                    #Empty current entries
                    self.current_min_entry.delete(0, tk.END)
                    self.current_max_entry.delete(0, tk.END)
                   
                    
                    self.current_min_entry.configure(state = 'disabled')
                    self.current_max_entry.configure(state = 'disabled')
                                      
            if self.factor_datatype == float:              
                self.current_dec_entry = self.dec_widgets[factor]
                
                if self.current_checkstate == True:
                    self.current_dec_entry.configure(state = 'normal')
                    
                elif self.current_checkstate == False:
                    self.current_dec_entry.delete(0, tk.END)
                    self.current_dec_entry.configure(state = 'disabled')
                    
          
            self.check_index += 1  