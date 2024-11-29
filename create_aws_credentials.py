import os

# Path to the credentials file
credentials_file = os.path.join(os.path.expanduser('~/.aws'), 'credentials')

# Create or modify the credentials file
def create_credentials(file_path):
    with open(file_path, 'w') as f:
        f.write('[default]\naws_access_key_id =AKIA47CRWXEM7CSHBI4J\naws_secret_access_key = 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\n')
    print(f"Credentials file created at {file_path}")

create_credentials(credentials_file)