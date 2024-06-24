#Import AWS Python SDK and JSON libraries
import boto3
import json

#Huggingface user access token
accessToken = 'XXXXXXXXXXXXXXXXXXXXXXXXXX'

#store HF token in secrets manager
secrets = boto3.client('secretsmanager', region_name='us-east-1')
try:
    response = secrets.create_secret(
    Name='HFaccessToken',
    SecretString= accessToken
    )
    secret_arn = response['ARN']
except secrets.exceptions.ResourceExistsException:
    #describe secret and store the arn as a variable
    response = secrets.describe_secret(
        SecretId='HFaccessToken')
    secret_arn = response['ARN']
    print("Secret already exists. Continuing...")

#Create IAM role for Ray nodes with Secrets Manager access
iam = boto3.client('iam')

# Define the policy name and policy document to access the HF secret
policy_name = 'ray-secret-access-policy'
policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "GetSecretPermissions",
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue",
                "secretsmanager:DescribeSecret"
            ],
            "Resource": f"'{secret_arn}'"
        }
    ]
}
policy_document_json = json.dumps(policy_document, ensure_ascii=False)

role_name = 'ray-head-role'

try:
    # Check if the role exists
    response = iam.get_role(RoleName=role_name)
    if 'Role' in response:
        ray_role_arn = response['Role']['Arn']
        print(f"Use the following IAM Role= {ray_role_arn}")
# Role doesn't exist, create it
except iam.exceptions.NoSuchEntityException:
    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument='''{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ec2.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }''',
    Description='Ray Head Role'
    )
    #Attach policies to IAM Role
    if 'Role' in response:
        role_arn = response['Role']['Arn']
        print(f"The IAM role '{role_name}' has been created.")
        print(f"Use the following IAM Role= {role_arn}")
    else:
        print("Failed to create the IAM role.")
try:
    #create instance profile
    instance_profile_response = iam.create_instance_profile(
        InstanceProfileName='ray-instance-profile')
    #Attach instance profile to the role
    iam.add_role_to_instance_profile(
        InstanceProfileName='ray-instance-profile',
        RoleName=role_name
    )
except iam.exceptions.EntityAlreadyExistsException:
        print("Instance profile already exists. Continuing...")
    
    # Create the IAM policy
try:
    response = iam.create_policy(
        PolicyName=policy_name,
        PolicyDocument=str(policy_document_json)
    )
    # Store the policy ARN as an output
    policy_arn = response['Policy']['Arn']
    policy_arns = [
        'arn:aws:iam::aws:policy/AmazonEC2FullAccess',
        'arn:aws:iam::aws:policy/AmazonS3FullAccess',
        policy_arn
    ]
    #Attach IAM policies to the role
    for policy_arn in policy_arns:
        try:
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"Attached policy {policy_arn} to role {role_name}")
        except Exception as e:
            print(f"Error attaching policy {policy_arn} to role {role_name}: {e}")

except iam.exceptions.EntityAlreadyExistsException:
    # If the policy already exists store the Arn
    print("IAM Policy already exists. Continuing...")
except Exception as e:
    print(f"An error occurred: {e}")