import os
import re

def parse_ag_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    points = []
    assumes = []
    proves = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split()
        if parts[0] == "point":
            points.append(parts[1])
        elif parts[0] == "assume":
            assumes.append(parts[1:])
        elif parts[0] == "prove":
            proves.append(parts[1:])
            
    return points, assumes, proves

def translate_to_newclid(points, assumes, proves):
    point_constructions = {p: [] for p in points}
    
    # Mapping logic for assumptions
    # In AG, the order of assumes usually follows the introduction of points.
    # We figure out the "target" point by finding the point in the assume arguments
    # that is being constrained. Typically, it's the latest point introduced among the arguments.
    
    for assume in assumes:
        pred = assume[0]
        args = assume[1:]
        
        # Find the target point (the one appearing latest in the points list)
        target = max(args, key=lambda p: points.index(p) if p in points else -1)
        others = [a for a in args if a != target]
        
        if pred == "coll":
            # coll A B C -> target is on_line of the other two
            point_constructions[target].append(f"on_line {target} {others[0]} {others[1]}")
        elif pred == "perp":
            # perp A B C D -> target A is on_tline through B perpendicular to C D
            if target == args[0]: # Target is A
                point_constructions[target].append(f"on_tline {target} {args[1]} {args[2]} {args[3]}")
            else:
                # Fallback, just use generic on_tline assuming order
                pass
        elif pred == "para":
            # para A B C D -> target A is on_pline through B parallel to C D
            if target == args[0]:
                point_constructions[target].append(f"on_pline {target} {args[1]} {args[2]} {args[3]}")
        elif pred == "cong":
            # cong A B C D -> B is on_circle centered at A with radius C D
            if len(args) == 4:
                # If target is B (args[1]) and A is args[0] and C=A, D=some point
                if target == args[1] and args[0] == args[2]:
                    point_constructions[target].append(f"on_circle {target} {args[0]} {args[3]}")
                elif target == args[3] and args[2] == args[0]:
                    point_constructions[target].append(f"on_circle {target} {args[2]} {args[1]}")
                else:
                    point_constructions[target].append(f"eqdistance {args[0]} {args[1]} {args[2]} {args[3]}")
        elif pred == "circle":
            pass # circle c1 a b c
        elif pred == "cyclic":
            pass
            
    # Format the JGEX string
    clauses = []
    # Find free points (first few points usually)
    free_points = []
    for p in points:
        if not point_constructions[p]:
            free_points.append(p)
        else:
            if free_points:
                # Group free points
                if len(free_points) == 2:
                    clauses.append(f"{free_points[0]} {free_points[1]} = segment {free_points[0]} {free_points[1]}")
                elif len(free_points) == 3:
                    clauses.append(f"{free_points[0]} {free_points[1]} {free_points[2]} = triangle {free_points[0]} {free_points[1]} {free_points[2]}")
                else:
                    for fp in free_points:
                        clauses.append(f"{fp} = free {fp}")
                free_points = []
            
            # Target with its constructions
            consts = ", ".join(point_constructions[p])
            clauses.append(f"{p} = {consts}")
            
    if free_points:
         for fp in free_points:
             clauses.append(f"{fp} = free {fp}")
             
    setup_str = "; ".join(clauses)
    
    goal_str = ""
    if proves:
        pred = proves[0][0]
        args = proves[0][1:]
        if pred == "cong": pred = "cong"
        elif pred == "eqangle": pred = "eqangle"
        goal_str = f" ? {pred} {' '.join(args)}"
        
    return setup_str + goal_str

def main():
    input_dir = "imo_ag_30"
    output_dir = "imo_translated"
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".txt"): continue
        if filename == "bpsa.txt": continue
        
        filepath = os.path.join(input_dir, filename)
        points, assumes, proves = parse_ag_file(filepath)
        translated = translate_to_newclid(points, assumes, proves)
        
        out_filepath = os.path.join(output_dir, filename)
        with open(out_filepath, "w") as f:
            f.write(translated)
            
        if "2000_p1" in filename:
            print("="*60)
            print("SANITY CHECK: IMO 2000 P1")
            print("="*60)
            print("[La nostra traduzione automatica]:")
            print(translated)
            print("\n[La traduzione ufficiale Newclid (da imo.py)]:")
            print("a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; m = on_circle m g1 a, on_circle m g2 b; n = on_circle n g1 a, on_circle n g2 b; c = on_pline c m a b, on_circle c g1 a; d = on_pline d m a b, on_circle d g2 b; p = on_line p a n, on_line p c d; q = on_line q b n, on_line q c d; e = on_line e a c, on_line e b d ? cong e p e q")
            print("="*60)

if __name__ == "__main__":
    main()
