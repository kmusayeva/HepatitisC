from utils import *

mpl.style.use('ggplot')

class DataVisualize:

    def __init__(self, dat):

        self.df = dat.getFinalData()
        self.diagnosis_dict = dat.diagnosis_dict
        self.colors_list = sns.color_palette('pastel')[:4]



    def getSummaryStatistics(self):

        print(self.df.describe())



    def Target(self):

        df_total = (self.df.groupby("Diagnosis_Multi")
                        .size()
                        .reset_index(name="Total")
                    )

        df_total.sort_values(by=['Diagnosis_Multi'],
                             key=lambda x: x.map(self.diagnosis_dict),
                             ascending=True,
                             inplace=True)


        plt.figure(figsize=(13, 5.5))

        df_total["Total"].plot(kind='pie',
                               autopct='%1.1f%%',
                               startangle=90,
                               shadow=None,
                               labels=None,
                               pctdistance=0.8,
                               colors=self.colors_list
                               )


        plt.title('', y=1.12, fontsize=15)

        plt.legend(labels=df_total['Diagnosis_Multi'], loc='upper left', fontsize=9)

        plt.tight_layout()

        plt.show()



    def AgeDisease(self):

        fig = plt.figure(figsize=(20, 5))

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # Histogram of Age
        self.df["Age"].plot.hist(bins=50, alpha=0.5, legend=False, xlabel='Age', ax=ax1)
        ax1.set_title("Age Distribution", y=1.12, fontsize=15)
        ax1.set_xlabel("Age")
        ax1.set_ylabel(None)

        # Age and Disease Binary
        sns.boxplot(x='Diagnosis_Binary',
                    y='Age',
                    data = self.df,
                    whis=1.1,
                    width=0.4,
                    palette="pastel",
                    hue='Diagnosis_Binary',
                    showfliers=False,
                    legend=None,
                    ax=ax2)

        sns.stripplot(x='Diagnosis_Binary',
                      y='Age',
                      data = self.df,
                      color='black',
                      alpha=0.3,
                      jitter=0.12,
                      ax=ax2)

        ax2.set_title("Age Distribution by Disease Presence", y=1.12, fontsize=15)
        ax2.set_ylabel("Age")
        ax2.set_xlabel(None)


        # Age and Disease Multi
        sns.boxplot(x='Diagnosis_Multi',
                    y='Age',
                    data = self.df,
                    whis=1.5,
                    width=0.4,
                    palette="pastel",
                    hue='Diagnosis_Multi',
                    showfliers=False, ax=ax3)

        sns.stripplot(x='Diagnosis_Multi',
                      y='Age',
                      data = self.df,
                      color='black',
                      alpha=0.2,
                      jitter=0.12,
                      ax=ax3)

        ax3.set_title("Age Distribution by Diagnosis Category", y=1.12, fontsize=15)
        ax3.set_ylabel("Age")
        ax3.set_xlabel(None)


        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()




    def GenderDisease(self):

        gender_counts = self.df['Sex'].value_counts().sort_index()

        gender_diagnosis = (self.df.groupby(['Sex', 'Diagnosis_Multi'])
                            .size()
                            .groupby(level=0)
                            .apply(lambda x: round(100 * x / x.sum(), 2))
                            .reset_index(level=0, drop=True)
                            )


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 4))

        # Plot 1: total number per gender
        gender_counts.plot(kind='bar', ax=ax1, rot=0, color="lightsteelblue")
        ax1.set_title("Total Observations per Gender", fontsize=14)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_xlabel("Gender", fontsize=12)

        for i, v in enumerate(gender_counts.values):
            ax1.text(i, v + max(gender_counts.values) * 0.01, str(v), ha='center', fontsize=11)


        # diagnosis distribution by gender
        gender_diagnosis.unstack().plot(kind='bar',
                                        ax=ax2,
                                        stacked=True,
                                        rot=0,
                                        color=self.colors_list
                                        )
        ax2.set_title("Disease Prevalence per Gender (Percent)", fontsize=14)
        ax2.set_ylabel("Percentage", fontsize=12)
        ax2.set_xlabel("Gender", fontsize=12)
        ax2.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.show()



    def Biomarkers(self, variable):

        fig = plt.figure(figsize=(19, 4))

        ax1 = fig.add_subplot(1, 4, 1)  # add subplot 1 (1 row, 3 columns, first plot)
        ax2 = fig.add_subplot(1, 4, 2)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        # Histogram of variable
        self.df[variable].plot.hist(bins=50, alpha=0.7, legend=False, color="blue", ax=ax1)
        ax1.set_title(f"{variable} Distribution", y=1.12, fontsize=15)
        ax1.set_xlabel("")
        ax1.set_ylabel(None)

        # Binary class-conditinal density plot of variable
        sns.kdeplot(data=self.df,
                    x=variable,
                    hue='Diagnosis_Multi',
                    common_norm=False,
                    fill=True,
                    alpha=0.4,
                    linewidth=1.5,
                    palette="muted",
                    ax=ax2)

        ax2.set_title(f"{variable} Distribution by Disease Presence",
                      y=1.12,
                      fontsize=14)
        ax2.set_ylabel(f"{variable}")
        ax2.set_xlabel(None)

        # Boxplot of variable conditioned on binary disease categories
        sns.boxplot(x='Diagnosis_Binary',
                    y=variable,
                    data=self.df,
                    whis=1.1,
                    width=0.6,
                    palette="muted",
                    hue='Diagnosis_Binary',
                    showfliers=False,
                    legend=None,
                    ax=ax3)

        sns.stripplot(x='Diagnosis_Binary',
                      y=variable,
                      data=self.df,
                      color='black',
                      alpha=0.5,
                      jitter=0.12,
                      size=4,
                      ax=ax3)


        ax3.set_title(f"{variable} Distribution by Disease Presence", y=1.12, fontsize=14)
        ax3.set_ylabel(f"{variable}")
        ax3.set_xlabel(None)

        # Boxplot of variable conditioned on multi disease categories
        sns.boxplot(x='Diagnosis_Multi',
                    y=variable,
                    data=self.df,
                    whis=1.5,
                    width=0.6,
                    palette="muted",
                    hue='Diagnosis_Multi',
                    showfliers=False, ax=ax4)

        sns.stripplot(x='Diagnosis_Multi',
                      y=variable,
                      data=self.df,
                      color='black',
                      alpha=0.5,
                      jitter=0.12,
                      size=4,
                      ax=ax4)


        ax4.set_title(f"{variable} Distribution by Diagnosis Category", y=1.12, fontsize=14)
        ax4.set_ylabel(f"{variable}")
        ax4.set_xlabel(None)
        plt.tight_layout()
        plt.subplots_adjust(top=0.7)
        plt.show()



    def missingness(self):

        df_missingness = (self.df.drop(columns=["Diagnosis", "Diagnosis_Binary"])
                          .groupby('Diagnosis_Multi')
                          .apply(lambda g: g.isna().sum(), include_groups=False)
                          )

        sns.heatmap(df_missingness, cmap="viridis", annot=True)
        plt.title("Proportion of Missing Values by Diagnosis_Label")
        plt.show()


    def BiomarkerSummary(self, var):
        bio_df = self.df[[var, "Diagnosis_Multi"]]
        return bio_df.groupby("Diagnosis_Multi").describe() \
            .sort_values(by=['Diagnosis_Multi'],
                         key=lambda x: x.map(self.diagnosis_dict),
                         ascending=True)


    