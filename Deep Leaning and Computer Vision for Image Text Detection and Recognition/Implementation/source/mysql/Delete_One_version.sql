set versionIDParam = 10;
-- Delete records from child table 'assessment_outcome'
DELETE FROM assessment_outcome 
WHERE assessmentID IN (SELECT assessmentID FROM assessment WHERE courseID IN (SELECT courseID FROM course WHERE versionID = versionIDParam));

-- Delete records from child table 'assessment'
DELETE FROM assessment 
WHERE courseID IN (SELECT courseID FROM course WHERE versionID = versionIDParam);

-- Delete records from child table 'outcome'
DELETE FROM outcome 
WHERE courseID IN (SELECT courseID FROM course WHERE versionID = versionIDParam);

-- Delete records from child table 'completion'
DELETE FROM completion 
WHERE courseID IN (SELECT courseID FROM course WHERE versionID = versionIDParam);

-- Delete records from child table 'prerequisite'
DELETE FROM prerequisite 
WHERE courseID IN (SELECT courseID FROM course WHERE versionID = versionIDParam);

-- Delete records from 'course' table
DELETE FROM course WHERE versionID = versionIDParam;

-- Delete records from 'version' table
DELETE FROM version WHERE versionID = versionIDParam;