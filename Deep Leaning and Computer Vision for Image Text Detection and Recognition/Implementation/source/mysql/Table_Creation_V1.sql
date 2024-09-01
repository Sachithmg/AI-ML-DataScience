DROP DATABASE IF EXISTS education_nz;
CREATE DATABASE education_nz; 
USE education_nz;

SET NAMES utf8 ;
SET character_set_client = utf8mb4 ;

CREATE TABLE version (
  versionID INTEGER(11) NOT NULL AUTO_INCREMENT, 
  name VARCHAR(50) NOT NULL, 
  description VARCHAR(1000) DEFAULT NULL, 
  created_datetime DATETIME NOT NULL,
  PRIMARY KEY (versionID)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE course (
  courseID INTEGER(11) NOT NULL AUTO_INCREMENT, 
  code VARCHAR(10) NOT NULL, 
  title VARCHAR(100) NOT NULL, 
  credit INTEGER(11) DEFAULT 0,
  level INTEGER(11) DEFAULT 0,
  self_directed_learning_hrs INTEGER(11) DEFAULT 0, 
  tutor_directed_learning_hrs INTEGER(11) DEFAULT 0, 
  aim VARCHAR(4000) DEFAULT NULL, 
  versionID INTEGER(11) NOT NULL,
  pageNo INTEGER(11) NOT NULL, 
  PRIMARY KEY (courseID),
  KEY FK_course_version_id (versionID),
  CONSTRAINT FK_course_version_id FOREIGN KEY (versionID) REFERENCES version (versionID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE prerequisite (
  prerequisiteID INTEGER(11) NOT NULL AUTO_INCREMENT, 
  code VARCHAR(10) NOT NULL, 
  title VARCHAR(100) NOT NULL, 
  courseID INTEGER(11) NOT NULL,
  prerequisite_courseID INTEGER(11) DEFAULT NULL, 
  PRIMARY KEY (prerequisiteID), 
  KEY FK_prerequisite_course_id (courseID),
  KEY FK_prerequisite_course_id2 (prerequisite_courseID),
  CONSTRAINT FK_prerequisite_course_id FOREIGN KEY (courseID) REFERENCES course (courseID) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT FK_prerequisite_course_id2 FOREIGN KEY (prerequisite_courseID) REFERENCES course (courseID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE completion (
  completionID INTEGER(11) NOT NULL AUTO_INCREMENT, 
  criteria VARCHAR(2000) NOT NULL, 
  courseID INTEGER(11) NOT NULL,
  PRIMARY KEY (completionID),
  KEY FK_completion_course_id (courseID),
  CONSTRAINT FK_completion_course_id FOREIGN KEY (courseID) REFERENCES course (courseID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE outcome (
  outcomeID INTEGER(11) NOT NULL AUTO_INCREMENT, 
  outcome_Index INTEGER(11) NOT NULL, 
  outcome VARCHAR(2000) NOT NULL, 
  courseID INTEGER(11) NOT NULL, 
  PRIMARY KEY (outcomeID), 
  KEY FK_outcome_course_id (courseID),
  CONSTRAINT FK_outcome_course_id FOREIGN KEY (courseID) REFERENCES course (courseID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE assessment (
  assessmentID INTEGER(11) NOT NULL AUTO_INCREMENT,
  method VARCHAR(1000) NOT NULL, 
  weight_precentage VARCHAR(5) DEFAULT NULL,
  courseID INTEGER(11) NOT NULL, 
  PRIMARY KEY (assessmentID), 
  KEY FK_assessment_course_id (courseID),
  CONSTRAINT FK_assessment_course_id FOREIGN KEY (courseID) REFERENCES course (courseID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE assessment_outcome (
  assessmentID INTEGER(11) NOT NULL, 
  outcomeID_value VARCHAR(5), 
  outcomeID INTEGER(11) DEFAULT NULL, 
  PRIMARY KEY ( assessmentID, outcomeID_value),
  KEY FK_assess_outcome_ID1 (assessmentID),
  KEY FK_assess_outcome_ID2 (outcomeID),
  CONSTRAINT FK_assess_outcome_ID1 FOREIGN KEY (assessmentID) REFERENCES assessment (assessmentID) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT FK_assess_outcome_ID2 FOREIGN KEY (outcomeID) REFERENCES outcome (outcomeID) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE error (
  errorID INTEGER(11) NOT NULL AUTO_INCREMENT, 
  title VARCHAR(100) NOT NULL, 
  description VARCHAR(2000) DEFAULT NULL, 
  creation_datetime datetime NOT NULL, 
  PRIMARY KEY (errorID)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

