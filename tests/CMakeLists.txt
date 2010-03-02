###############################################
# Project Testing
###############################################
	SET (TEST_COLLECTION
#		helloworld      # WORKS
#		message_test	# WORKS
#		density1	# WORKS
#		density2	# WORKS
#		density3	# WORKS
#		bollig_main	# INCOMPLETE
		original	# WORKS
		rbf_message	# INCOMPLETE
		template	# WORKS
	)

	FOREACH (test ${TEST_COLLECTION})
		ADD_SUBDIRECTORY (${test})
	ENDFOREACH (test)
