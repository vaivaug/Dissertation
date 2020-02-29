from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as auc_plt


def plot_AUC(test_OUTPUT, prediction_probs):

    # no skill prediction
    no_skill_probs = [0 for _ in range(len(test_OUTPUT))]

    # calculate scores
    no_skill_auc = roc_auc_score(test_OUTPUT, no_skill_probs)
    model_auc = roc_auc_score(test_OUTPUT, prediction_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % no_skill_auc)
    print('Logistic: ROC AUC=%.3f' % model_auc)

    # calculate roc curves
    no_skills_false_pos_rate, no_skill_true_pos_rate, thresholds1 = roc_curve(test_OUTPUT, no_skill_probs)
    model_false_positive_rate, model_true_pos_rate, thresholds2 = roc_curve(test_OUTPUT, prediction_probs)

    # plot the roc curve for the model
    auc_plt.plot(no_skills_false_pos_rate, no_skill_true_pos_rate, linestyle='--', label='No skills')
    auc_plt.plot(model_false_positive_rate, model_true_pos_rate, marker='.', label='Logistic')

    # axis labels
    auc_plt.xlabel('False Positive Rate')
    auc_plt.ylabel('True Positive Rate')
    # show the legend
    auc_plt.legend()
    # show the plot
    auc_plt.show()
    auc_plt.savefig('plots/auc_plt.png')
    auc_plt.clf()



