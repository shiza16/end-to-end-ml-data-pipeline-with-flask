import sys

def error_message_detail(error, error_detail:str):
    _,_,exec_tb = error_detail.exc_infor()
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message = "Errored occured in python script name [{0}] line number [{1}] error message [{2}]".format(
    file_name, exec_tb.tb_lineno,str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_details)

    def __str__(self):
        return self.error_message