import subprocess

def generate_requirements(filename='requirements.txt'):
    try:
        with open(filename, 'w') as f:
            subprocess.run(['pip', 'freeze'], stdout=f, check=True)
        print(f"✅ '{filename}' created successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error running pip freeze:", e)
    except Exception as e:
        print("❌ Unexpected error:", e)

if __name__ == "__main__":
    generate_requirements()
