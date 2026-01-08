import json
import copy

def update_notebook():
    notebook_path = 'visualizeResult.ipynb'
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print(f"Error: {notebook_path} not found.")
        return

    cells = nb['cells']
    
    # Identify key cells
    data_loading_cell = None
    plotting_cell = None
    
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "result_folder = 'result'" in source:
                data_loading_cell = cell
                # Assume the next cell is the plotting cell
                if i + 1 < len(cells) and cells[i+1]['cell_type'] == 'code':
                    plotting_cell = cells[i+1]
                break
    
    if not data_loading_cell:
        print("Could not find data loading cell.")
        return

    new_experiments = [
        ('result-1024x1024', 'Experiment 1: Standard Resolution'),
        ('result-1024x1024-upscale', 'Experiment 2: Upscale Resolution')
    ]

    for result_dir, title in new_experiments:
        # Create markdown header
        header_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n"]
        }
        nb['cells'].append(header_cell)

        # Create data loading cell
        new_data_cell = copy.deepcopy(data_loading_cell)
        new_data_cell['execution_count'] = None
        new_data_cell['outputs'] = []
        # Replace result folder
        new_source = []
        for line in new_data_cell['source']:
            if "result_folder = 'result'" in line:
                new_source.append(f"result_folder = '{result_dir}'\n")
            else:
                new_source.append(line)
        new_data_cell['source'] = new_source
        nb['cells'].append(new_data_cell)

        # Create plotting cell
        if plotting_cell:
            new_plot_cell = copy.deepcopy(plotting_cell)
            new_plot_cell['execution_count'] = None
            new_plot_cell['outputs'] = []
            nb['cells'].append(new_plot_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print(f"Successfully updated {notebook_path}")

if __name__ == "__main__":
    update_notebook()
