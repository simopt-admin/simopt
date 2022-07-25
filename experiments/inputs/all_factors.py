# all_solver_fixed_factors = {"RNDSRCH10": {},
#                             "RNDSRCH20": {},
#                             "RNDSRCH30": {},
#                             "RNDSRCH40": {},
#                             "RNDSRCH50": {}
#                             } #{"sample_size": 20}}

# all_problem_fixed_factors = {"CNTNEWS-1": {},
#                              "FACSIZE-2": {},
#                              "RMITD-1": {},
#                              "SSCONT-1":{}
#                              }

# all_model_fixed_factors = {"CNTNEWS-1": {},
#                             "FACSIZE-2": {},
#                             "RMITD-1": {},
#                             "SSCONT-1":{}
#                              }
                            
all_solver_fixed_factors = {"RNDSRCHss20": {"sample_size": 20},
                            "RNDSRCHss30": {"sample_size": 30}
                            }

all_problem_fixed_factors = {"MM1-1mu3": {"budget": 100},
                             "MM1-1mu4": {"budget": 100},
                             }

all_model_fixed_factors = {"MM1-1mu3": {"mu": 3.0},
                             "MM1-1mu4": {"mu": 4.0},
                            }
                        
# mymeta = MetaExperiment(solver_names=["RNDSRCH","RNDSRCH"], problem_names=["MM1-1","MM1-1"], solver_renames=["RNDSRCHss20","RNDSRCHss30"], problem_renames=["MM1-1mu3","MM1-1mu4"], fixed_factors_filename="all_factors")
# mymeta.run(n_macroreps=10)
# mymeta.post_replicate(n_postreps=20, crn_across_budget=True, crn_across_macroreps=False)