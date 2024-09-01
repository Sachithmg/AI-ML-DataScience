SELECT 
    comparison_versionID,


    -- Ratio
    AVG(code_ratio) AS avg_code_ratio,
    AVG(title_ratio) AS avg_title_ratio,
    AVG(credit_ratio) AS avg_credit_ratio,
    AVG(level_ratio) AS avg_level_ratio,
    AVG(self_directed_learning_ratio) AS avg_self_directed_learning_ratio,
    AVG(tutor_directed_learning_ratio) AS avg_tutor_directed_learning_ratio,
    AVG(aim_ratio) AS avg_aim_ratio,
    AVG(completion_ratio) AS avg_completion_ratio,
    AVG(outcome_ratio) AS avg_outcome_ratio,
    AVG(prerequisite_code_ratio) AS avg_prerequisite_code_ratio,
    AVG(prerequisite_title_ratio) AS avg_prerequisite_title_ratio,
    AVG(assessment_method_ratio) AS avg_assessment_method_ratio,
    AVG(assessment_weight_ratio) AS avg_assessment_weight_ratio,
    AVG(assessment_outcome_ratio) AS avg_assessment_outcome_ratio


FROM course_comparison
where comparison_versionID = 9
GROUP BY comparison_versionID;