/* 
Legende: 
https://raw.githubusercontent.com/lbynum/machine-bias-criminal-risk/master/compas_report.pdf
*/

SELECT 
COUNT(*) 
-- Auswahl der Spalten:
/*Allg. Infos:  name, */ id,  sex, race, age,  marital_status,
/*Jugendstrafen: */ juv_fel_count, juv_misd_count, juv_other_count,
/*Anazahl vorheriger Vergehen:*/ priors_count,  
/*Rückfällig? */ is_recid, is_violent_recid,
/*Ursprünglicher Fall: */ c_custody_in_days,
/*Ursprünglicher Fall - Beschreibung: */ c_charges, c_charge_degrees,
/*Nachfolge Verbrechen*/ r_custody_in_days,
/*Nachfolge Verbrechen - Beschreibung: */ r_charges, r_charge_degrees
/*Haftzeit in Tagen aus Prison History:*/ , h_prison,
/*Haftzeit in Tagen aus Jail History:*/ h_jail,
/*Compas:*/ rec_supervision_level, raw_v, score_v_txt, dec_v, raw_r, score_r_txt, dec_r, raw_a, score_a_txt, dec_a --compas_screening_date
-- Drei Typen von Assassement: Risk of Violence (risk_v), Risk of Recidivsm (risk_r), Risk of Failure to Appear (risk_a)

FROM 
	(
	SELECT *
	FROM
		(
		SELECT *,
			-- Haftzeit in Tagen des Eintrages c_jail in PEOPLE
			ROUND (CAST (julianday(c_jail_out) - julianday(c_jail_in) AS REAL )) As c_custody_in_days,
			-- Haftzeit in Tagen des Eintrages r_jail in PEOPLE
			ROUND (CAST (julianday(r_jail_out) - julianday(r_jail_in) AS REAL )) As r_custody_in_days
		FROM people
		) 
	AS peopl
	
	-- Informationen aus CHARGE zum c_case
	LEFT JOIN 
		(
			SELECT case_number,
			-- Zusammenfassen aller aufgelisteten Delikte der entsprechenden Fallnummer (c_case)
				group_concat (charge) AS c_charges, 
				group_concat (charge_degree) AS c_charge_degrees
			FROM charge
			GROUP BY case_number
		) AS A
	ON peopl.c_case_number = A.case_number

	-- Informationen aus CHARGE zum r_case
	LEFT JOIN 
		( -- Zusammenfassen aller aufgelisteten Delikte der entsprechenden Fallnummer (r_case)
			SELECT case_number,
				group_concat (charge) AS r_charges, 
				group_concat (charge_degree) AS r_charge_degrees
			FROM charge
			GROUP BY case_number
		) AS B
	ON peopl.r_case_number = B.case_number

	-- Prisonhistory
	LEFT JOIN 
		(
		SELECT *
		FROM (
			SELECT person_id, sum(custody_time) over (PARTITION By person_id) as h_prison 
			FROM (
				SELECT person_id, 
				-- Ausgabe in Tage; < 12h Aufenthalt resultieren in 0 Tagen
					ROUND (CAST (julianday(out_custody) - julianday(in_custody) AS REAL )) As custody_time
				FROM prisonhistory
				)
		GROUP BY person_id
			)
		) AS ph 
	ON peopl.id = ph.person_id

	-- Jailhistory, analog zu Prisonhistory
	LEFT JOIN 
		(
		SELECT *
		FROM (
			SELECT person_id, sum(custody_time) over (PARTITION By person_id) as h_jail 
			FROM (
				SELECT person_id, 
					ROUND (CAST (julianday(out_custody) - julianday(in_custody) AS REAL )) As custody_time
				FROM jailhistory
				)
		GROUP BY person_id
			)
		) AS jh 
	ON peopl.id = jh.person_id

	-- COMPAS-Daten. Zu Jedem COMPAS-Termin existieren mindest 3 Einträge (Risk of... ). In 131 Fällen existieren für einen Compas-Termin > 2 Einträge. 
	LEFT JOIN
		( 
			(
			SELECT DISTINCT person_id, marital_status, rec_supervision_level, raw_score AS raw_v, score_text as score_v_txt, decile_score AS dec_v, screening_date 
			FROM compas
			WHERE type_of_assessment = 'Risk of Violence'
			) AS risk_1
			
			LEFT JOIN 
				(
				SELECT DISTINCT person_id, raw_score AS raw_r, score_text as score_r_txt, decile_score AS dec_r, screening_date 
				FROM compas
				WHERE type_of_assessment = 'Risk of Recidivism'
				) AS risk_2 ON (risk_1.person_id = risk_2.person_id AND risk_1.screening_date = risk_2.screening_date)
				
				LEFT JOIN 
					(
					SELECT DISTINCT person_id, raw_score AS raw_a, score_text as score_a_txt, decile_score AS dec_a, screening_date
					FROM compas
					WHERE type_of_assessment = 'Risk of Failure to Appear'
					) AS risk_3 ON (risk_1.person_id = risk_3.person_id AND risk_1.screening_date = risk_3.screening_date)				
		)	
		AS risk 
	ON (peopl.compas_screening_date = risk.screening_date AND peopl.id = risk.person_id)
)

-- Herausfiltern aller Personen die an einem Tag zwei COMPAS-Tests mit unterschiedlichen Ergebnissen (Insgesamt 131 Einträge, z.B. 
GROUP BY ID
-- ORDER BY ID
HAVING COUNT(id) = 1
