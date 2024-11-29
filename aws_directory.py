import os

aws_dir = os.path.expanduser('~/.aws')

# Create the .aws directory if it doesn't exist
if not os.path.exists(aws_dir):
    os.makedirs(aws_dir)
    print(f"Created directory: '{aws_dir}'")
else:
    print(f"Directory already exists: '{aws_dir}'")

# Set permissions for the .aws directory
os.chmod(aws_dir, 0o700)
print("Permissions set to '700' for .aws directory.")