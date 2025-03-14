import xml.etree.ElementTree as ET
import os

def remap_query_numbers(input_xml_path, output_xml_path):
    try:
        # Explicitly print and verify paths
        print("--- Path Verification ---")
        print(f"Script Location: {os.path.abspath(__file__)}")
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Input XML Path: {input_xml_path}")
        print(f"Output XML Path: {output_xml_path}")
        print("-------------------------")

        # Check if input file exists
        if not os.path.exists(input_xml_path):
            print(f"ERROR: Input file {input_xml_path} does not exist!")
            return None

        # Load and parse the XML file
        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        # Count queries before modification
        original_count = len(root.findall('top'))
        print(f"\nFound {original_count} queries in input file")

        # Update query numbers sequentially
        for index, top in enumerate(root.findall('top'), start=1):
            num = top.find('num')
            if num is not None:
                num.text = f" {index} "  # Consistent spacing
            else:
                print(f"Warning: Query {index} has no num element")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)

        # Save the updated XML
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

        # Verify the output
        print(f"\nSuccessfully renumbered {original_count} queries")
        print(f"Output saved to: {output_xml_path}")

        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Construct full paths
    input_xml_path = os.path.join(project_root, "data", "cran.qry.xml")
    output_xml_path = os.path.join(project_root, "data", "cran_queries_renumbered.xml")
    
    print("Starting query renumbering...")
    result = remap_query_numbers(input_xml_path, output_xml_path)
    
    if result:
        print("Query renumbering completed successfully!")
    else:
        print("Query renumbering failed.")
        exit(1)