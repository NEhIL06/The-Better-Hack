import singlestoredb as s2
import pandas as pd
import numpy as np

# Function to create a connection to SingleStoreDB
def create_connection():
    return s2.connect("string")


# Insert embeddings into SingleStoreDB
try:
    with create_connection() as conn:
        with conn.cursor() as cur:
            # Create a table to store embeddings if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS food (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    food TEXT,
                    description TEXT,
                    embedding VECTOR(1024, F32) NOT NULL
                )
            """)

            # Read data from XLSX
            df = pd.read_excel('embedded_output_ollama.xlsx')  # Replace with your XLSX file path

            for _, row in df.iterrows():
                food = row[0]
                description = row[1]
                embedding = np.fromstring(row[2], sep=',').tolist()  # Convert string to numpy array and then to list

                # Format embedding as a JSON array string
                embedding_json = f"[{','.join(map(str, embedding))}]"

                # Insert into the table
                cur.execute("""
                    INSERT INTO food (food, description, embedding) VALUES (%s, %s, %s)
                """, (food, description, embedding_json))  # Pass the JSON array string

            # Commit the transaction
            conn.commit()
            print("Insertion successful.")
except Exception as e:
    print(f"Insertion failed: {e}")
