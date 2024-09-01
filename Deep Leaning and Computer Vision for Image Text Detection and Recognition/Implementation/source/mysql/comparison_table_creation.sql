
USE education_nz;

SET NAMES utf8 ;
SET character_set_client = utf8mb4 ;

CREATE TABLE comparison_version (
  comparison_versionID INTEGER NOT NULL AUTO_INCREMENT, 
  comparison_name VARCHAR(50) NOT NULL, 
  comparison_description VARCHAR(1000) DEFAULT NULL, 
  comparison_version_1 INTEGER NOT NULL, 
  comparison_version_2 INTEGER NOT NULL,
  created_datetime DATETIME NOT NULL,
  PRIMARY KEY (comparison_versionID)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE course_comparison (
  course_comparison_ID INTEGER NOT NULL AUTO_INCREMENT, 
  comparison_versionID INTEGER NOT NULL,
  pageNo INTEGER NOT NULL, 
  code_ratio DECIMAL(5, 2) DEFAULT NULL,
  code_version1 TEXT DEFAULT NULL, 
  code_version2 TEXT DEFAULT NULL, 
  title_ratio DECIMAL(5, 2) DEFAULT NULL,
  title_version1 TEXT DEFAULT NULL, 
  title_version2 TEXT DEFAULT NULL,
  credit_ratio DECIMAL(5, 2) DEFAULT NULL,
  credit_version1 TEXT DEFAULT NULL, 
  credit_version2 TEXT DEFAULT NULL,
  level_ratio DECIMAL(5, 2) DEFAULT NULL,
  level_version1 TEXT DEFAULT NULL, 
  level_version2 TEXT DEFAULT NULL,
  self_directed_learning_ratio DECIMAL(5, 2) DEFAULT NULL,
  self_directed_learning_version1 TEXT DEFAULT NULL, 
  self_directed_learning_version2 TEXT DEFAULT NULL,
  tutor_directed_learning_ratio DECIMAL(5, 2) DEFAULT NULL,
  tutor_directed_learning_version1 TEXT DEFAULT NULL, 
  tutor_directed_learning_version2 TEXT DEFAULT NULL,
  aim_ratio DECIMAL(5, 2) DEFAULT NULL,
  aim_version1 TEXT DEFAULT NULL, 
  aim_version2 TEXT DEFAULT NULL,
  completion_ratio DECIMAL(5, 2) DEFAULT NULL,
  completion_version1 TEXT DEFAULT NULL, 
  completion_version2 TEXT DEFAULT NULL,
  outcome_ratio DECIMAL(5, 2) DEFAULT NULL,
  outcome_version1 TEXT DEFAULT NULL, 
  outcome_version2 TEXT DEFAULT NULL,
  prerequisite_code_ratio DECIMAL(5, 2) DEFAULT NULL,
  prerequisite_code_version1 TEXT DEFAULT NULL, 
  prerequisite_code_version2 TEXT DEFAULT NULL,
  prerequisite_title_ratio DECIMAL(5, 2) DEFAULT NULL,
  prerequisite_title_version1 TEXT DEFAULT NULL, 
  prerequisite_title_version2 TEXT DEFAULT NULL,
  assessment_method_ratio DECIMAL(5, 2) DEFAULT NULL,
  assessment_method_version1 TEXT DEFAULT NULL, 
  assessment_method_version2 TEXT DEFAULT NULL,
  assessment_weight_ratio DECIMAL(5, 2) DEFAULT NULL,
  assessment_weight_version1 TEXT DEFAULT NULL, 
  assessment_weight_version2 TEXT DEFAULT NULL,
  assessment_outcome_ratio DECIMAL(5, 2) DEFAULT NULL,
  assessment_outcome_version1 TEXT DEFAULT NULL, 
  assessment_outcome_version2 TEXT DEFAULT NULL,
  PRIMARY KEY (course_comparison_ID),
  KEY FK_course_com_version_id (comparison_versionID),
  CONSTRAINT FK_course_com_version_id FOREIGN KEY (comparison_versionID) REFERENCES comparison_version (comparison_versionID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;