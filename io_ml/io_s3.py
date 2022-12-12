import boto3
from retry import retry
import os
from io_ml.io_ml import IO_ML
from utils.logger import Logger

"""
Classe que trata entrada e saída no S3 da AWS
"""
class IO_S3(IO_ML):
    
    """
    Construtor
    @param acc_key_name Nome da chave para o S3 (variável de ambiente do Fury)
    @param sec_ken_name Chave para o S3 (variável de ambiente do Fury)
    @param local_path Diretório local do arquivo
    @param Bucket to S3
    @param remote_path Diretório remoto onde arquivo será inserido
    """
    def __init__(self,acc_key_name, sec_key_name, local_path, bucket, remote_path):
        self.logger = Logger(self)
        
        self.local_path = local_path
        self.bucket = bucket
        self.remote_path = remote_path
        
        AWS_KEY = os.environ[acc_key_name] # colocar os nomes das chaves do secrets
        AWS_SECRET = os.environ[sec_key_name]
        self.s3 = boto3.Session(aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)
        self.s3client = self.s3.client('s3')
    
    """
    Lê arquivo do S3 e faz download
    """
    def read(self):
        
        self.logger.log('Download de {remote} para {local}'.format(local=self.local_path,
                                                                 remote=self.remote_path))
        self.s3client.download_file(self.bucket, 
                                  self.remote_path, 
                                  self.local_path)
    """
    Lê diretório do S3 e faz download
    """
    def read_folder(self):

        s3 = self.s3.resource("s3")
        bucket = s3.Bucket(self.bucket)
        
        for obj in bucket.objects.filter(Prefix=self.remote_path):
            
            target = obj.key if self.local_path is None \
                else os.path.join(self.local_path, os.path.relpath(obj.key, self.remote_path))
            
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            
            if obj.key[-1] == '/':
                continue
            
            bucket.download_file(obj.key, target)
    
    """
    Escreve arquivo no S3
    """
    def write(self):      
        
        self.logger.log('Upload de {local} para {remote}'.format(local=self.local_path,
                                                                 remote=self.remote_path))
        
        self.s3client.upload_file(self.local_path, 
                                 self.bucket, 
                                 self.remote_path)