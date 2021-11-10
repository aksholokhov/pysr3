Search.setIndex({docnames:["community_guidelines","index","models_overview","quickstart","source/modules","source/pysr3","source/pysr3.glms","source/pysr3.glms.link_functions","source/pysr3.glms.models","source/pysr3.glms.oracles","source/pysr3.glms.problems","source/pysr3.linear","source/pysr3.linear.models","source/pysr3.linear.oracles","source/pysr3.linear.problems","source/pysr3.lme","source/pysr3.lme.model_selectors","source/pysr3.lme.models","source/pysr3.lme.oracles","source/pysr3.lme.priors","source/pysr3.lme.problems","source/pysr3.logger","source/pysr3.preprocessors","source/pysr3.priors","source/pysr3.regularizers","source/pysr3.solvers"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["community_guidelines.rst","index.rst","models_overview.rst","quickstart.rst","source/modules.rst","source/pysr3.rst","source/pysr3.glms.rst","source/pysr3.glms.link_functions.rst","source/pysr3.glms.models.rst","source/pysr3.glms.oracles.rst","source/pysr3.glms.problems.rst","source/pysr3.linear.rst","source/pysr3.linear.models.rst","source/pysr3.linear.oracles.rst","source/pysr3.linear.problems.rst","source/pysr3.lme.rst","source/pysr3.lme.model_selectors.rst","source/pysr3.lme.models.rst","source/pysr3.lme.oracles.rst","source/pysr3.lme.priors.rst","source/pysr3.lme.problems.rst","source/pysr3.logger.rst","source/pysr3.preprocessors.rst","source/pysr3.priors.rst","source/pysr3.regularizers.rst","source/pysr3.solvers.rst"],objects:{"":[[5,0,0,"-","pysr3"]],"pysr3.glms":[[7,0,0,"-","link_functions"],[8,0,0,"-","models"],[9,0,0,"-","oracles"],[10,0,0,"-","problems"]],"pysr3.glms.link_functions":[[7,1,1,"","IdentityLinkFunction"],[7,1,1,"","LinkFunction"],[7,1,1,"","PoissonLinkFunction"]],"pysr3.glms.link_functions.IdentityLinkFunction":[[7,2,1,"","gradient"],[7,2,1,"","hessian"],[7,2,1,"","value"]],"pysr3.glms.link_functions.LinkFunction":[[7,2,1,"","gradient"],[7,2,1,"","hessian"],[7,2,1,"","value"]],"pysr3.glms.link_functions.PoissonLinkFunction":[[7,2,1,"","gradient"],[7,2,1,"","hessian"],[7,2,1,"","value"]],"pysr3.glms.models":[[8,1,1,"","PoissonL1Model"],[8,1,1,"","PoissonL1ModelSR3"],[8,1,1,"","SimplePoissonModel"],[8,1,1,"","SimplePoissonModelSR3"]],"pysr3.glms.models.PoissonL1Model":[[8,2,1,"","instantiate"]],"pysr3.glms.models.PoissonL1ModelSR3":[[8,2,1,"","instantiate"]],"pysr3.glms.models.SimplePoissonModel":[[8,2,1,"","get_information_criterion"],[8,2,1,"","instantiate"],[8,2,1,"","predict_problem"]],"pysr3.glms.models.SimplePoissonModelSR3":[[8,2,1,"","get_information_criterion"],[8,2,1,"","instantiate"],[8,2,1,"","predict_problem"]],"pysr3.glms.oracles":[[9,1,1,"","GLMOracle"],[9,1,1,"","GLMOracleSR3"]],"pysr3.glms.oracles.GLMOracle":[[9,2,1,"","aic"],[9,2,1,"","bic"],[9,2,1,"","forget"],[9,2,1,"","gradient"],[9,2,1,"","gradient_value_function"],[9,2,1,"","hessian"],[9,2,1,"","instantiate"],[9,2,1,"","loss"],[9,2,1,"","value_function"]],"pysr3.glms.oracles.GLMOracleSR3":[[9,2,1,"","aic"],[9,2,1,"","bic"],[9,2,1,"","find_optimal_parameters"],[9,2,1,"","forget"],[9,2,1,"","gradient_value_function"],[9,2,1,"","gradient_x"],[9,2,1,"","instantiate"],[9,2,1,"","loss"],[9,2,1,"","value_function"]],"pysr3.glms.problems":[[10,1,1,"","PoissonProblem"]],"pysr3.glms.problems.PoissonProblem":[[10,2,1,"","from_dataframe"],[10,2,1,"","from_x_y"],[10,2,1,"","generate"]],"pysr3.linear":[[12,0,0,"-","models"],[13,0,0,"-","oracles"],[14,0,0,"-","problems"]],"pysr3.linear.models":[[12,1,1,"","LinearCADModel"],[12,1,1,"","LinearCADModelSR3"],[12,1,1,"","LinearL1Model"],[12,1,1,"","LinearL1ModelSR3"],[12,1,1,"","LinearModel"],[12,1,1,"","LinearSCADModel"],[12,1,1,"","LinearSCADModelSR3"],[12,1,1,"","SimpleLinearModel"],[12,1,1,"","SimpleLinearModelSR3"]],"pysr3.linear.models.LinearCADModel":[[12,2,1,"","instantiate"]],"pysr3.linear.models.LinearCADModelSR3":[[12,2,1,"","instantiate"]],"pysr3.linear.models.LinearL1Model":[[12,2,1,"","instantiate"]],"pysr3.linear.models.LinearL1ModelSR3":[[12,2,1,"","instantiate"]],"pysr3.linear.models.LinearModel":[[12,2,1,"","check_is_fitted"],[12,2,1,"","fit"],[12,2,1,"","fit_problem"],[12,2,1,"","instantiate"],[12,2,1,"","predict"],[12,2,1,"","predict_problem"]],"pysr3.linear.models.LinearSCADModel":[[12,2,1,"","instantiate"]],"pysr3.linear.models.LinearSCADModelSR3":[[12,2,1,"","instantiate"]],"pysr3.linear.models.SimpleLinearModel":[[12,2,1,"","get_information_criterion"],[12,2,1,"","instantiate"]],"pysr3.linear.models.SimpleLinearModelSR3":[[12,2,1,"","get_information_criterion"],[12,2,1,"","instantiate"]],"pysr3.linear.oracles":[[13,1,1,"","LinearOracle"],[13,1,1,"","LinearOracleSR3"]],"pysr3.linear.oracles.LinearOracle":[[13,2,1,"","aic"],[13,2,1,"","bic"],[13,2,1,"","forget"],[13,2,1,"","gradient"],[13,2,1,"","gradient_value_function"],[13,2,1,"","hessian"],[13,2,1,"","instantiate"],[13,2,1,"","loss"],[13,2,1,"","value_function"]],"pysr3.linear.oracles.LinearOracleSR3":[[13,2,1,"","aic"],[13,2,1,"","bic"],[13,2,1,"","find_optimal_parameters"],[13,2,1,"","forget"],[13,2,1,"","gradient_value_function"],[13,2,1,"","instantiate"],[13,2,1,"","loss"],[13,2,1,"","value_function"]],"pysr3.linear.problems":[[14,1,1,"","LinearProblem"]],"pysr3.linear.problems.LinearProblem":[[14,2,1,"","from_dataframe"],[14,2,1,"","from_x_y"],[14,2,1,"","generate"],[14,2,1,"","to_x_y"]],"pysr3.lme":[[16,0,0,"-","model_selectors"],[17,0,0,"-","models"],[18,0,0,"-","oracles"],[19,0,0,"-","priors"],[20,0,0,"-","problems"]],"pysr3.lme.model_selectors":[[16,3,1,"","get_model"],[16,3,1,"","select_covariates"]],"pysr3.lme.models":[[17,1,1,"","CADLmeModel"],[17,1,1,"","CADLmeModelSR3"],[17,1,1,"","L0LmeModel"],[17,1,1,"","L0LmeModelSR3"],[17,1,1,"","L1LmeModel"],[17,1,1,"","L1LmeModelSR3"],[17,1,1,"","LMEModel"],[17,1,1,"","SCADLmeModel"],[17,1,1,"","SCADLmeModelSR3"],[17,1,1,"","SimpleLMEModel"],[17,1,1,"","SimpleLMEModelSR3"]],"pysr3.lme.models.CADLmeModel":[[17,2,1,"","instantiate"]],"pysr3.lme.models.CADLmeModelSR3":[[17,2,1,"","instantiate"]],"pysr3.lme.models.L0LmeModel":[[17,2,1,"","instantiate"]],"pysr3.lme.models.L0LmeModelSR3":[[17,2,1,"","instantiate"]],"pysr3.lme.models.L1LmeModel":[[17,2,1,"","instantiate"]],"pysr3.lme.models.L1LmeModelSR3":[[17,2,1,"","instantiate"]],"pysr3.lme.models.LMEModel":[[17,2,1,"","check_is_fitted"],[17,2,1,"","fit"],[17,2,1,"","fit_problem"],[17,2,1,"","instantiate"],[17,2,1,"","predict"],[17,2,1,"","predict_problem"],[17,2,1,"","score"]],"pysr3.lme.models.SCADLmeModel":[[17,2,1,"","instantiate"]],"pysr3.lme.models.SCADLmeModelSR3":[[17,2,1,"","instantiate"]],"pysr3.lme.models.SimpleLMEModel":[[17,2,1,"","get_information_criterion"],[17,2,1,"","instantiate"]],"pysr3.lme.models.SimpleLMEModelSR3":[[17,2,1,"","get_information_criterion"],[17,2,1,"","instantiate"]],"pysr3.lme.oracles":[[18,1,1,"","LinearLMEOracle"],[18,1,1,"","LinearLMEOracleSR3"]],"pysr3.lme.oracles.LinearLMEOracle":[[18,2,1,"","beta_gamma_to_x"],[18,2,1,"","demarginalized_loss"],[18,2,1,"","flip_probabilities_beta"],[18,2,1,"","forget"],[18,2,1,"","get_condition_numbers"],[18,2,1,"","get_ic"],[18,2,1,"","gradient_beta"],[18,2,1,"","gradient_gamma"],[18,2,1,"","gradient_value_function"],[18,2,1,"","hessian_beta"],[18,2,1,"","hessian_beta_gamma"],[18,2,1,"","hessian_gamma"],[18,2,1,"","instantiate"],[18,2,1,"","joint_gradient"],[18,2,1,"","joint_loss"],[18,2,1,"","jones2010bic"],[18,2,1,"","loss"],[18,2,1,"","muller_hui_2016ic"],[18,2,1,"","optimal_beta"],[18,2,1,"","optimal_gamma"],[18,2,1,"","optimal_gamma_ip"],[18,2,1,"","optimal_gamma_pgd"],[18,2,1,"","optimal_obs_std"],[18,2,1,"","optimal_random_effects"],[18,2,1,"","vaida2005aic"],[18,2,1,"","value_function"],[18,2,1,"","x_to_beta_gamma"]],"pysr3.lme.oracles.LinearLMEOracleSR3":[[18,2,1,"","find_optimal_parameters"],[18,2,1,"","find_optimal_parameters_ip"],[18,2,1,"","gradient_beta"],[18,2,1,"","gradient_gamma"],[18,2,1,"","gradient_value_function"],[18,2,1,"","hessian_beta"],[18,2,1,"","hessian_gamma"],[18,2,1,"","joint_gradient"],[18,2,1,"","joint_loss"],[18,2,1,"","jones2010bic"],[18,2,1,"","loss"],[18,2,1,"","muller_hui_2016ic"],[18,2,1,"","optimal_beta"],[18,2,1,"","vaida2005aic"],[18,2,1,"","value_function"]],"pysr3.lme.priors":[[19,1,1,"","GaussianPriorLME"],[19,1,1,"","NonInformativePriorLME"]],"pysr3.lme.priors.GaussianPriorLME":[[19,2,1,"","forget"],[19,2,1,"","gradient_beta"],[19,2,1,"","gradient_gamma"],[19,2,1,"","hessian_beta"],[19,2,1,"","hessian_beta_gamma"],[19,2,1,"","hessian_gamma"],[19,2,1,"","instantiate"],[19,2,1,"","loss"]],"pysr3.lme.priors.NonInformativePriorLME":[[19,2,1,"","forget"],[19,2,1,"","gradient_beta"],[19,2,1,"","gradient_gamma"],[19,2,1,"","hessian_beta"],[19,2,1,"","hessian_beta_gamma"],[19,2,1,"","hessian_gamma"],[19,2,1,"","instantiate"],[19,2,1,"","loss"]],"pysr3.lme.problems":[[20,1,1,"","LMEProblem"],[20,1,1,"","LMEStratifiedShuffleSplit"],[20,1,1,"","Problem"],[20,3,1,"","get_per_group_coefficients"],[20,3,1,"","random_effects_to_matrix"]],"pysr3.lme.problems.LMEProblem":[[20,2,1,"","from_dataframe"],[20,2,1,"","from_x_y"],[20,2,1,"","generate"],[20,2,1,"","to_x_y"]],"pysr3.lme.problems.LMEStratifiedShuffleSplit":[[20,2,1,"","get_n_splits"],[20,2,1,"","split"]],"pysr3.lme.problems.Problem":[[20,2,1,"","from_x_y"],[20,2,1,"","to_x_y"]],"pysr3.logger":[[21,1,1,"","Logger"]],"pysr3.logger.Logger":[[21,2,1,"","add"],[21,2,1,"","append"],[21,2,1,"","get"],[21,2,1,"","log"]],"pysr3.preprocessors":[[22,1,1,"","Preprocessor"]],"pysr3.preprocessors.Preprocessor":[[22,2,1,"","add_intercept"],[22,2,1,"","normalize"]],"pysr3.priors":[[23,1,1,"","GaussianPrior"],[23,1,1,"","NonInformativePrior"],[23,1,1,"","Prior"]],"pysr3.priors.GaussianPrior":[[23,2,1,"","forget"],[23,2,1,"","gradient"],[23,2,1,"","hessian"],[23,2,1,"","instantiate"],[23,2,1,"","loss"]],"pysr3.priors.NonInformativePrior":[[23,2,1,"","forget"],[23,2,1,"","gradient"],[23,2,1,"","hessian"],[23,2,1,"","instantiate"],[23,2,1,"","loss"]],"pysr3.regularizers":[[24,1,1,"","CADRegularizer"],[24,1,1,"","DummyRegularizer"],[24,1,1,"","ElasticRegularizer"],[24,1,1,"","L0Regularizer"],[24,1,1,"","L1Regularizer"],[24,1,1,"","PositiveQuadrantRegularizer"],[24,1,1,"","PositiveQuadrantRegularizerLME"],[24,1,1,"","Regularizer"],[24,1,1,"","SCADRegularizer"]],"pysr3.regularizers.CADRegularizer":[[24,2,1,"","forget"],[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.DummyRegularizer":[[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.ElasticRegularizer":[[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.L0Regularizer":[[24,2,1,"","forget"],[24,2,1,"","instantiate"],[24,2,1,"","optimal_tbeta"],[24,2,1,"","optimal_tgamma"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.L1Regularizer":[[24,2,1,"","forget"],[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.PositiveQuadrantRegularizer":[[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.PositiveQuadrantRegularizerLME":[[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.Regularizer":[[24,2,1,"","forget"],[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.regularizers.SCADRegularizer":[[24,2,1,"","forget"],[24,2,1,"","instantiate"],[24,2,1,"","prox"],[24,2,1,"","value"]],"pysr3.solvers":[[25,1,1,"","FakePGDSolver"],[25,1,1,"","PGDSolver"]],"pysr3.solvers.FakePGDSolver":[[25,2,1,"","optimize"]],"pysr3.solvers.PGDSolver":[[25,2,1,"","optimize"]],pysr3:[[6,0,0,"-","glms"],[11,0,0,"-","linear"],[15,0,0,"-","lme"],[21,0,0,"-","logger"],[22,0,0,"-","preprocessors"],[23,0,0,"-","priors"],[24,0,0,"-","regularizers"],[25,0,0,"-","solvers"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[2,3,8,9,10,12,13,14,17,18,19,20,21,24,25],"0001":[9,13,17,25],"01621459":18,"0428727350273315":3,"05":[8,12,17],"06":18,"1":[2,3,8,9,10,12,13,14,17,18,19,20,24,25],"10":[3,10,14,18,20],"100":[2,10,14],"1000":[8,9,12,13,17,20,25],"10000":17,"1080":18,"1215989":18,"1326":18,"15":3,"15055187290939537":3,"1e":[2,3,8,12,17,18],"1e2":3,"1e3":3,"2":[2,3,12,17,18,19],"20":3,"200":[2,18],"2005":17,"2010":[17,18],"2016":[17,18],"2018":18,"21":3,"24":3,"25":20,"2673485":18,"3":[2,3,12,17,18,20],"300":3,"33":3,"4":[2,3,20],"40":17,"42":[3,10,14,20],"5":[2,3,9,17,20],"50":3,"500":3,"51536734_bayesian_information_criterion_for_longitudinal_and_clustered_data":18,"55":3,"6":3,"7":[2,3,12,17],"9":3,"\u00b2_\ud835\udec4":18,"\u03b2":[17,18,20],"\u03bb":[17,18],"\u2112":18,"\ud835\udca9":[17,18,20],"\ud835\udec4":[17,18,20],"\ud835\udf3a_i":[17,18,20],"_\ud835\udec4":18,"abstract":20,"break":0,"byte":7,"case":[8,12,17,20],"class":[2,3,7,8,9,10,12,13,14,17,18,19,20,21,22,23,24,25],"default":[2,12,16,17,20],"do":[17,18],"final":20,"float":[7,8,12,17,18,19,20,23,24,25],"function":[2,3,8,12,16,17,18,24],"import":[2,3,16],"int":[7,8,12,17,18,20,24,25],"new":[8,12,17],"public":18,"return":[8,12,16,17,18,19,20,21,23,24,25],"static":[10,14,18,20,22,23],"t\u03b2":[17,18],"t\u03c9_ix_i":18,"t\u03c9_iy_i":18,"t\ud835\udec4":[17,18],"throw":[8,12,17],"true":[3,9,12,17,18,20,24],A:[1,3,17],As:0,For:[0,2,17],If:[8,12,17,18,20,21,24,25],In:[2,18,20,24],It:[1,2,3,8,12,17,18,20,24,25],NOT:20,No:2,Not:[12,17,18],The:[2,3,17,18,19,20],There:[17,20],To:0,_0:17,_1:2,_:[2,23],_array_lik:7,_dont_solve_wrt_beta:18,_i:2,_supportsarrai:7,ab:3,about:24,abov:[12,17,19,20,23],absolut:[2,18,24],access:25,accord:20,account:18,accur:[1,2,3],accuraci:17,accuracy_scor:3,activ:3,active_categorical_set:20,adapt:[1,24],add:[17,20,21],add_intercept:[10,14,22],addit:[18,21,24],address:2,adher:0,adjust:18,advic:2,affect:17,against:2,aic:[2,3,9,13,17,18],akaik:18,aksholokhov:0,alasso_model:2,algorithm:[2,8,12,17,18],all:[2,12,16,17,18,19,20,21,23,24],allow:[17,25],almost:18,alpha:[2,8,12,24],alreadi:21,also:[3,20,25],alwai:[17,25],amplitud:[12,17,18],an:[0,2,3,8,12,17,18,20,24,25],analyt:25,ani:[0,7,18,20,21],answer:[12,17,18,20],anyth:20,append:21,appli:[17,18],applic:24,ar:[2,3,12,16,17,18,20,21,25],arbitrarili:17,arg:[18,19],argument:[18,20,24],around:18,arrai:[3,12,17,18,20],array_lik:17,as_x_i:20,assum:18,attach:[18,24],automat:[3,16],avail:[2,3,18],ax:2,b:[3,10,14,20,24],bar:2,base:[2,7,8,9,10,12,13,14,17,18,19,20,21,22,23,24,25],baseestim:[12,17],basic:2,bay:18,becaus:17,befor:[8,12,17],begin:20,being:[18,24],below:[0,2,3],best:[3,17],best_estimator_:3,best_model:3,best_params_:3,beta:[3,17,18,19,20,23,24],beta_gamma_to_x:18,between:20,bic:[2,3,8,9,12,13,17,18],big:2,bigger:[2,8,12],black:16,bool:[7,8,12,17,18,20,24],both:[2,3,17,18,20],bound:20,box:16,broader:0,bug:0,build:20,c:[2,10,14],cad:[1,3,12,16,17,24],cad_sr3:16,cadlmemodel:[2,17],cadlmemodelsr3:[2,17],cadregular:[2,24],calcul:[17,18],call:[8,12,17,18],can:[0,2,3,8,12,16,17,18,20,24,25],candid:16,categor:20,cd:0,chanc:20,chance_miss:20,chance_outli:20,chang:2,characterist:20,check:[12,17],check_estim:0,check_is_fit:[12,17],child:18,choic:[2,3],choos:[2,3],chose:3,clf:3,clip:[2,24],clone:0,cluster:20,code:0,coef_:[2,3],coeffici:[2,3,12,17,18,20],column:[3,12,16,17,20,23],column_label:20,columns_label:[3,17,20],com:[0,18],comment:0,commun:1,compat:[0,3,20],complex:7,compon:3,comput:[0,18],condit:[2,18],conduct:0,confusion_matrix:3,connect:2,consid:18,constant:[8,12,17,24],constraint:[8,9],construct:24,constructor:16,contain:[3,8,12,16,17,20],content:4,contribut:0,contributor:0,control:[2,17,20],conveni:3,converg:[2,8,12,17,25],convert:[3,20],coordin:[18,24],core:[2,10,14,16,20],correct:3,correctli:3,correspond:[8,12,16,17,20],cov:16,covari:[16,17,18,20,24],coven:0,creat:[3,17,18,19,20,23,24,25],criteria:18,criterion:[2,3,8,12,17,18,25],cross:[2,3],cumul:20,current:[1,2,3,12,17],custom:[2,24],cut:[12,17],cv:3,data:[3,8,10,12,14,16,17,18,20],datafram:[10,14,16,20],dataset:[3,20,23],de:18,debug:[8,12,17,18],defin:[17,20],definit:18,degre:2,demarginalized_loss:18,denomin:2,depend:[3,12,17,19,23,24],deriv:20,descent:[2,17,18,25],describ:3,design:[1,3,17,18,25],desir:24,detach:18,detail:[3,17,18],determin:17,deviat:[2,17,20,24],df:16,diag:[17,18,20],diagnost:16,dict:[12,16,17,19,20,21,23],dictionari:[16,20,21],direct:[8,12],disabl:20,disregard:17,distribut:[20,23],divis:20,do_correction_step:[9,18],doc:18,docstr:0,document:[3,17],doe:[0,2,17,18,20],doi:18,don:0,dostr:0,dot:3,dtype:7,dummyregular:24,dure:[3,18],e:[2,17,19,25],each:[3,18,20],effect1:20,effect:[1,8,12,16,17,18,19,20,23,24],effects2:20,effici:[8,12],either:[8,12,17,25],el:[8,12],elastic_ep:17,elasticregular:24,element:[3,17,23,24],ell:[2,17],em:17,empow:[2,3],empti:16,encompass:24,encourag:0,ensur:0,entiti:17,enumer:2,ep:24,equal:20,equat:18,error:[3,8,12,17,18,20],estim:[3,12,17,18,24],estimator_check:0,etc:[24,25],evalu:[3,8,12,18,19,23,24],everi:[2,3,20,25],exampl:[2,3],exclus:18,expect:17,expens:17,experi:18,experiment:18,extens:0,extra:[3,18],f:[2,3],fake:24,fakepgdsolv:25,fals:[8,9,10,12,13,14,17,18,20,24],familiar:3,faster:[2,17],fe_column:20,fe_param:19,fe_regularization_weight:[17,20],featur:[1,2,3,10,14,16,17,20],features_covariance_matrix:20,features_label:[3,20],ffn:3,ffp:3,field:[12,17],fig:2,figur:16,file:[16,20],find_optimal_paramet:[9,13,18],find_optimal_parameters_ip:18,fine:17,finit:18,first:[12,17,20,24],fit:[2,3,8,12,17,19],fit_fixed_intercept:[17,20],fit_problem:[12,17],fit_random_intercept:[17,20],five:[2,18],fix:[0,3,8,12,16,17,18,19,20,23,24,25],fixed_effect:[16,20],fixed_featur:20,fixed_step_len:[8,12,17,25],flat:2,flip:18,flip_probabilities_beta:18,fn:3,folder:16,follow:[0,3,17],forget:[9,13,18,19,23,24],form:[2,8,12,17,18,20],format:[8,12,17,19,20,23],found:[2,3],fourth:3,fp:3,frac:2,fraction:20,frame:[10,14,16,20],framework:2,frequent:18,from:[2,3,17,18,20,24,25],from_datafram:[10,14,20],from_x_i:[10,14,20],ftn:3,ftp:3,full:18,fulli:3,funciton:2,futur:[18,20],g:[17,19,25],gamma:[3,17,18,19,20,23,24],gaussian:[19,23],gaussianprior:[19,23],gaussianpriorlm:19,gener:[2,3,10,14,18,20,25],generator_param:20,get:[17,20,21],get_condition_numb:18,get_ic:18,get_information_criterion:[3,8,12,17],get_model:16,get_n_split:20,get_per_group_coeffici:20,git:0,github:0,give:2,given:[12,17,18,20,24],glm:[4,5],glmoracl:9,glmoraclesr3:9,global:20,glossari:3,go:[3,20],goal:2,got:3,grad_gamma:18,gradient:[2,7,9,13,17,18,19,23,25],gradient_beta:[18,19],gradient_gamma:[18,19],gradient_value_funct:[9,13,18],gradient_x:9,grid:[2,3,16],group1:20,group2:20,group:[3,8,12,16,17,20],group_label:20,groups_siz:[3,20],guarante:20,guidelin:1,ha:[2,3,20,24],handl:20,hard:2,hardwar:0,have:[3,8,12,17,18,20],helper:[20,21],here:[3,12,17,18,20],hessian:[7,9,13,18,19,23],hessian_beta:[18,19],hessian_beta_gamma:[18,19],hessian_gamma:[18,19],higher:3,how:[2,3,17,18,25],http:[0,18],hui:17,hyper:2,i:[2,18],ic:[3,8,12,17,18],ic_jones2010b:18,ic_vaida2005a:18,ident:20,identifi:3,identitylinkfunct:7,ignor:21,illustr:2,implement:[2,3,16,17,18,19,20,23,24,25],impli:24,impos:2,improv:2,includ:20,inclus:20,increas:[18,20],increase_lambda:18,independ:20,independent_beta_and_gamma:24,index:1,individu:24,info:[8,12,17,18],inform:[2,3,8,12,17,18,19,23,24],initi:[8,12,17,18,20,21,25],initial_paramet:[12,17],innov:2,input:17,insid:[18,25],instal:[0,1],instanc:[8,12,17,18,20,25],instanti:[8,9,12,13,17,18,19,23,24],instead:[2,18],intercept:[17,19,20],intercept_label:20,interfac:0,interior:18,intern:[17,18,20,25],interpol:2,invers:18,invok:0,ip:18,irrelev:3,issu:2,item:2,iter:[2,8,12,17,18,20,21,25],its:[2,18,20],itself:25,j:17,joint_gradi:18,joint_loss:18,jone:[17,18],jones2010b:18,jones_b:17,jstor:18,k:[2,17,18,20,24],kei:[8,12,16,17,21],kernel:18,knot:[2,12,17,24],know:[8,12,17],known:2,kwarg:[8,9,12,13,16,17,18,19,20,24,25],l0:[1,2,3,16,17,24],l0_sr3:16,l0lmemodel:[2,17],l0lmemodelsr3:[2,17],l0regular:24,l1:[1,16,24],l1_sr3:16,l1lmemodel:[2,17],l1lmemodelsr3:[2,3,17],l1regular:[2,24],l:[8,12,17],label:[12,17,20],lam:[2,3,9,13,17,24],lambda:[2,3,24],landscap:2,larger:[2,24],largest:24,lasso:[1,3,8,12,17,24],latent:20,lb:[17,18],lead:2,left:[2,12,17,18,20],len:20,length:[17,20,25],less:[2,25],level:[16,17],lg:[17,18],librari:2,like:[2,20,25],likelihood:[2,18],line:[8,12,17,18,25],line_search:18,linear:[1,4,5,8,9,10,16,17,18,20,22],linearcadmodel:[2,12],linearcadmodelsr3:[2,12],linearl1model:[2,12],linearl1modelsr3:[2,3,12],linearlmeoracl:[17,18,24,25],linearlmeoraclesr3:18,linearlmeproblem:[17,18],linearlmesparsemodel:[12,17],linearmodel:[2,12],linearoracl:[12,13],linearoraclesr3:13,linearproblem:[2,3,9,10,12,13,14,22],linearscadmodel:[2,12],linearscadmodelsr3:[2,12],link_funct:[5,6,9],linkfunct:[7,9],linspac:2,lipschitz:[8,12,17],list:[0,3,8,10,12,14,16,17,20,21,23],list_of_kei:21,lme:[3,4,5,24,25],lmemodel:[2,17],lmeproblem:[3,8,16,17,18,19,20,23],lmestratifiedshufflesplit:[3,20],locat:2,log:[12,18,21,25],log_progress:18,logger:[4,5,8,9,12,17,18,25],logger_kei:[8,12,17],loguniform:3,loss:[2,9,13,17,18,19,20,23,24,25],m:[18,20],magnitud:[2,20],mai:[8,12,17],make:[3,8,12,17,25],mani:3,map:2,margin:18,match:23,matplotlib:2,matric:[18,20],matrix:[12,17,18,20],max:20,max_it:[9,13,25],max_iter_oracl:17,max_iter_solv:[8,12,17],maxim:[8,12,17,18,25],maximum:25,mayb:20,maybe_beta:3,maybe_gamma:3,maybe_x:3,mean:[17,19,22],measur:20,merg:18,method:[1,2,3,8,12,17,18,21,25],methodolog:2,metric:3,min:20,minim:[2,18,24],minimum:25,miss:20,mix:[1,16,17,18,19,20],mode:0,model:[1,5,6,11,15,16,18,19,20,23],model_nam:16,model_select:3,model_selector:[5,15],modern:2,modul:[1,4],more:[2,3,17,18,24],most:[16,17],mu_decai:9,much:2,mueller:18,muller2018:3,muller:[17,18],muller_:[3,17],muller_hui_2016:18,multipl:17,multipli:20,must_include_f:20,must_include_featur:[10,14],must_include_r:20,n:[2,3,12,18,20,24],n_iter:3,n_iter_inn:18,n_job:3,n_split:[3,20],nabla:2,name:[2,16,19,20,23],nb:20,ncol:2,ndarrai:[8,12,17,18,19,20,23,24,25],nearli:2,necessari:16,need:[20,24],neg:17,net:18,never:20,nnz:17,nnz_beta:24,nnz_gamma:24,nnz_tbeta:[17,18,24],nnz_tgamma:[17,24],nois:18,non:[2,3,8,12,17,18,19,23,24,25],non_regularized_coeffici:2,none:[8,9,10,12,13,14,16,17,18,19,20,22,23,24,25],noninformativeprior:[19,23],noninformativepriorlm:19,norm:25,normal:[2,3,18,20,22],not_regularized_f:20,not_regularized_r:20,notfittederror:[12,17],noth:16,notic:2,now:20,np:[2,3,8,12,17,18,20,24],nrow:2,num_featur:[2,3,10,14],num_object:[2,3,10,14],number:[0,3,8,12,17,18,20,24,25],numer:[2,17,18,25],numpi:[0,2,3,7,12,17,18,20,24],object:[3,7,9,13,14,17,18,19,20,21,22,23,24,25],obs_std:[10,14],obs_var:[3,17,20],observ:[16,18,20],off:[12,17],often:[17,25],one:[3,17,18,20],ones:[12,17,25],onli:[2,8,12,17,18,20,24],oper:[2,24,25],optim:[2,18,25],optimal_beta:18,optimal_gamma:18,optimal_gamma_ip:18,optimal_gamma_pgd:18,optimal_obs_std:18,optimal_random_effect:18,optimal_tbeta:24,optimal_tgamma:24,option:[2,3,8,9,10,12,13,14,16,17,18,20,24,25],oracl:[5,6,11,12,15,17,24,25],order:2,order_of_object:20,org:18,origin:[17,25],other:[3,16,20,24],other_regular:24,otherwis:[8,12,17,20],out:3,outer:18,outlier:20,outlier_multipli:20,output:16,output_fold:16,over:[3,20],overrid:[12,17],overview:1,own:20,p:[3,17,18,20],packag:[1,3,4],page:[1,18],pair:21,panda:[3,10,14,16,20],paper:17,param:[3,23],param_distribut:3,paramet:[2,3,8,12,16,17,18,19,20,21,23,24,25],pars:16,part:[2,18,20,25],participation_in_select:17,particular:23,pass:[8,12,17,18],path:16,pathlib:16,pd:[16,20],penal:[2,12,17,20,24],penalti:2,per:20,per_group_coeffici:20,perform:[3,18],person:0,pgd:[1,8,12,17,18,25],pgdsolver:[12,17,25],pictur:2,pip:3,pleas:0,pledg:0,plot:2,plt:2,plu:20,point:[2,3,18,24,25],poissonl1model:8,poissonl1modelsr3:8,poissonlinkfunct:7,poissonproblem:10,polici:[8,12,17,25],popular:[2,3],portion:3,posit:[12,17,18],positivequadrantregular:24,positivequadrantregularizerlm:24,possibl:[8,12,17],posterior:18,practic:[8,9,12,13,17],practicalsr3:25,pre:17,pre_sel_cov:16,predict:[8,12,17],predict_problem:[8,12,17],preprocessor:[4,5],present:3,preserv:24,prevent:24,previou:[12,17],print:3,prior:[3,4,5,8,9,12,13,15,17,18],probabl:18,problem:[2,3,5,6,8,9,11,12,13,15,16,17,18,19,22,23,24,25],problem_column:23,progress:[18,21,25],project:[17,18,24],promot:[2,20],provid:[2,3,12,17,18,24,25],prox:[2,24],prox_step_len:18,proxim:[2,18,24,25],psd:20,pull:0,purpos:[18,25],py:0,pyplot:2,pysr3:2,pytest:0,python:[0,3],pyyaml:3,q:[17,18],quantiti:[19,23],quickstart:[1,2],r2_score:17,r:[2,17,19,23,24],rais:20,randn:3,random:[3,16,17,18,19,20,23,24],random_effect:[16,20],random_effects_to_matrix:20,random_featur:20,random_st:[3,20],randomizedsearchcv:3,randomli:20,rang:3,ravel:3,re:3,re_column:20,re_param:19,re_regularization_weight:[17,20],real:[12,17],record:21,regress:[2,12,17,18],regressormixin:[12,17],regular:[1,3,4,5,8,9,12,13,17,18,20,25],regularization_weight:[2,10,12,14],relax:[1,2,3,8,12,17,18,25],releas:[19,23],relev:20,represent:[3,20],request:[0,17,18],requir:[0,1,2],researchg:18,residu:17,respect:[17,18,19,20,23,24],rest:[21,24],result:[16,18,24],return_true_model_coeffici:20,rfn:3,rfp:3,rho:[2,12,17,24],role:[3,20],routin:[16,18,25],row:[17,20],rtn:3,rtp:3,rule:0,s:[0,2,3,12,17,18,20,21,24,25],safeguard:2,same:[2,8,12,16,17,19,20,23],sampl:[3,17],sample_weight:17,scad:[1,3,12,16,17,24],scad_sr3:16,scadlmemodel:[2,17],scadlmemodelsr3:[2,17],scadregular:[2,24],scalar:18,scikit_learn:3,scipi:3,score:[3,17],search:[1,2,3,8,12,17,18,25],second:[12,17,20,24],see:[2,3,17,18,20],seed:[3,10,14,20],select:[1,2,3,16,20],select_covari:16,selector:3,self:[12,17,21,23],semant:0,sens:18,separ:18,sequenc:7,sequenti:18,set:[2,3,8,12,17,18,20,21,24],set_titl:2,setup:0,shall:[17,20],shape:[3,12,17,18,20,24],should:[8,12,17,18,20,25],show:[2,3],shown:3,shrink:25,shuffl:20,sigma:[2,12,17,18,24],sign:18,significantli:20,simpl:[2,3,17],simplelinearmodel:[2,8,12],simplelinearmodelsr3:[2,8,12],simplelmemodel:[2,17],simplelmemodelsr3:[2,17],simplepoissonmodel:8,simplepoissonmodelsr3:8,simultan:[2,3,18],singl:17,situat:25,size:[3,8,12,17,18,20,25],sklearn:[0,3,12,17],skmix:[18,20],smooth:[2,8,12,17,25],smoothli:24,so:[3,20],solut:[2,17,25],solv:[17,25],solver:[2,4,5,8,12,17,18],some:20,space:3,spars:[2,17,18,24],sparser:2,sparsiti:[2,3,16,20],special:0,specif:20,specifi:[16,17],spline:[2,12,17,24],split:[2,3,18,20],sqrt:3,squar:17,sr3:[1,3,8,12,17,18],sr3practic:18,stabl:18,stack:20,standard:[0,3,17,20],start:[0,17,18,21,25],statist:17,std:[17,18,19,20,22],step:[8,12,17,18,24,25],stop:[8,12,17,18,25],store:[16,21],str:[7,8,10,12,14,16,17,18,19,20,21,23,25],stratifi:20,strength:[2,3,8,12,17,24],structur:16,study_id:16,submit:0,submodul:[2,4],subpackag:4,subplot:2,subroutin:[17,18,25],subspac:24,suitabl:[2,16],sum:[2,3,17],sum_:2,support:[1,3],suppos:12,sure:3,symmetr:20,t:[0,2,17,19,20,23,24],tail:18,take:[2,16,18,24],take_only_positive_part:18,tandfonlin:18,target:[10,14,16,20],tb:17,tbeta:[18,24],templat:[20,24],test:[1,3,20],test_siz:[3,20],text:2,tgamma:[18,24],than:[2,24,25],them:[2,3],thi:[0,2,8,12,17,18,20,21,25],those:3,three:2,threshold:[2,18],tight_layout:2,tighter:[2,8,12],time:[19,20],tn:3,to_x_i:[2,3,14,20],togeth:25,tol:[9,13,25],tol_inn:18,tol_oracl:17,tol_solv:[8,12,17],toler:[8,12,17,18,25],too:[2,24],top:18,total:17,tp:3,track:[8,12,17],transform:20,treat:17,true_group_label:20,true_paramet:[3,20],true_random_effect:20,true_rms:20,true_x:[2,3,10,14],tune:17,tupl:[12,17,18,19,20,23],tutori:2,two:2,txt:3,type:[7,24],u:[17,18,20],u_i:[17,18,20],ultim:20,under:18,uniform:20,union:[7,16,20],unlink:24,updat:[17,18,25],update_prox_everi:[17,18,25],us:[0,2,3,8,12,16,17,18,20,25],usag:1,usual:[16,17],util:[0,3],v:17,vaida2005a:18,vaida:[17,18],vaida_a:17,valid:[2,3],valu:[2,7,8,12,16,17,18,19,20,21,23,24,25],value_funct:[9,13,18],valueerror:20,vari:3,variabl:[17,18,20,24,25],varianc:[3,16,17,18,19,20,23],variou:[16,19,20,23,24],vector:[3,12,17,18,19,20,23,24],veri:[2,3],version:[0,2],vertic:20,via:[2,3],vs:1,w:[9,13,18,19,23,24],wa:[8,12,17],warm:18,warm_start:[12,17,18],warm_start_du:18,warm_start_oracl:17,we:[0,2,3,24],weight:[2,17,24],well:[2,3,16,20,24],were:20,what:21,whatev:24,when:[2,12,16,17,18,20,24,25],where:[17,18,20,24,25],whether:[8,12,17,18,20],which:[8,12,16,17,18,20,21,24],within:20,without:2,work:[2,17,24],wors:17,would:[3,17],wrapper:18,www:18,x0:[9,12,13,25],x:[2,3,7,8,9,10,12,13,14,17,18,20,23,24,25],x_:2,x_i:[2,17,18,20],x_k:2,x_test:20,x_to_beta_gamma:18,x_train:20,xb:20,xi:2,y:[2,3,8,10,12,14,17,20],y_i:[17,18,20],y_pred:17,y_test:20,y_train:20,y_true:17,yaml:16,yield:3,you:[0,3,8,12,17],your:[0,2,8,12,17],z:18,z_i:[17,18,20],zero:[2,3,17,18,24]},titles:["Community Guidelines","Welcome to PySR3 documentation!","Models Overview","Quickstart with <code class=\"docutils literal notranslate\"><span class=\"pre\">pysr3</span></code>","pysr3","pysr3 package","pysr3.glms package","pysr3.glms.link_functions module","pysr3.glms.models module","pysr3.glms.oracles module","pysr3.glms.problems module","pysr3.linear package","pysr3.linear.models module","pysr3.linear.oracles module","pysr3.linear.problems module","pysr3.lme package","pysr3.lme.model_selectors module","pysr3.lme.models module","pysr3.lme.oracles module","pysr3.lme.priors module","pysr3.lme.problems module","pysr3.logger module","pysr3.preprocessors module","pysr3.priors module","pysr3.regularizers module","pysr3.solvers module"],titleterms:{A:2,adapt:2,cad:2,commun:0,content:[5,6,11,15],develop:[0,1],document:1,effect:[2,3],get:1,glm:[6,7,8,9,10],guidelin:0,indic:1,instal:3,l1:2,lasso:2,linear:[2,3,11,12,13,14],link_funct:7,lme:[15,16,17,18,19,20],logger:21,mix:[2,3],model:[2,3,8,12,17],model_selector:16,modul:[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],oracl:[9,13,18],overview:2,packag:[5,6,11,15],pgd:2,preprocessor:22,prior:[19,23],problem:[10,14,20],pysr3:[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],quickstart:3,regular:[2,24],requir:3,scad:2,solver:25,sr3:2,start:1,submodul:[5,6,11,15],subpackag:5,tabl:[1,2],test:0,usag:3,vs:2,welcom:1}})