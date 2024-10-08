import paramiko
import os
import time
from sys import argv

FILENAMES = ['stb_image.h', 'stb_image_write.h', 'stb_ds.h', 'vector.c', 'random.c', 'load_obj.h', 'load_obj.c', 'interval.c', 'aabb.c', 'texture.c', 'material.c', 'primitive.c', 'lbvh_node.c', 'lbvh.c', 'render.c', 'main.c', 'Makefile']

def send_files_via_scp(hostname, port, username, password, local_dir, remote_dir, files):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, password)
    
    sftp = ssh_client.open_sftp()
    for f in files:
      sftp.put(f'{local_dir}/{f}', f'{remote_dir}/{f}')
    sftp.close()
    ssh_client.close()

def execute_ssh_command(hostname, port, username, password, command):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, password)

    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode("utf-8")
    error = stderr.read().decode("utf-8")

    ssh_client.close()
    return output, error

def receive_file_via_scp(hostname, port, username, password, remote_file, local_path):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, password)
    
    sftp = ssh_client.open_sftp()
    sftp.get(remote_file, local_path)
    sftp.close()
    ssh_client.close()

def check_rendering_file(hostname, port, username, password, REMOTE_DIR):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, password)
    
    sftp = ssh_client.open_sftp()
    while True:
        if "rendering" not in sftp.listdir(REMOTE_DIR):
            break
        time.sleep(1)
    sftp.close()
    ssh_client.close()

def dispatch(secrets_file):
    hostname, port, username, password, DIR, REMOTE_DIR, COMPILER_PATH = open(secrets_file).read().split('\n')
    port = int(port)

    # Send files via SCP
    send_files_via_scp(hostname, port, username, password, DIR, REMOTE_DIR, FILENAMES)
    print(f'Transferred {FILENAMES} over')
    
    # Execute SSH command to make and run the file
    print('Building executable...')
    command = f'cd {REMOTE_DIR} && PATH={COMPILER_PATH}:/usr/bin make gpu'
    out, err = execute_ssh_command(hostname, port, username, password, command)
    if 'error' in err.lower():
        print(err)
        print('Error encountered while building executable, aborting!')
        return

    print('Running executable...')
    command = f'cd {REMOTE_DIR} && ./gpu'
    out, err = execute_ssh_command(hostname, port, username, password, command)

    if err:
        print(err)
        print('Error encountered while running executable, aborting!')
        return

    print(out)
    
    # Check for the presence of "rendering" file
    check_rendering_file(hostname, port, username, password, REMOTE_DIR)
    print('GPU has finished rendering')
    
    # Once "rendering" is no longer present, receive the result file via SCP
    receive_file_via_scp(hostname, port, username, password, f'{REMOTE_DIR}/output.png', f'{DIR}/output.png')
    print('Received output.png from server')

if __name__ == "__main__":
    if argv[1] == 'tch': dispatch('.secrets-tch')
    elif argv[1] == 'nus': dispatch('.secrets-nus')
