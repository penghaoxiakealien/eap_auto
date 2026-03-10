
import re

def validate_highlighted_output(text):
    """检查每条句子中 << >> 的数量是否与 'Number of important tokens' 匹配"""
    pattern = r"Number of important tokens: (\d+)\s+(.+?: .+?)\s+indirect object:"
    blocks = re.findall(pattern, text)
    print(f"Blocks found: {blocks}")
    for expected_str, sentence in blocks:
        expected = int(expected_str)
        actual = len(re.findall(r"<<[^<>]+?>>", sentence))
        print(f"Expected: {expected}, Actual: {actual} in: {sentence}")
        if expected != actual:
            print(f"Mismatch found: expected {expected}, got {actual} in: {sentence}")
            return False
    return True

input="""
Number of important tokens: 1  
1_test: After the rally, Max and <<Kyle>> went to the plaza. Max gave a flyer to {{Kyle}}  
indirect object: Kyle  

Number of important tokens: 1  
2_test: Then, Eva and <<Joseph>> had a deep discussion. Afterwards Eva said to {{Joseph}}  
indirect object: Joseph  

Number of important tokens: 1  
3_test: While planning, <<Kyle>> and Scott were at the conference. Scott gave a schedule to {{Kyle}}  
indirect object: Kyle  

Number of important tokens: 1  
4_test: After the festival, <<Lewis>> and Rachel went to the market. Lewis gave a cookie to {{Rachel}}  
indirect object: Rachel  

Number of important tokens: 1  
5_test: After the gathering, Maria and <<William>> went to the hall. Maria gave a book to {{William}}  
indirect object: William  

Number of important tokens: 1  
6_test: While organizing, Anna and <<Brian>> were at the venue. Brian gave a ticket to {{Anna}}  
indirect object: Anna  

Number of important tokens: 1  
7_test: When Jennifer and <<Simon>> got a pin at the museum, Simon decided to give it to {{Jennifer}}  
indirect object: Jennifer  

Number of important tokens: 1  
8_test: In the garden, <<Sam>> and Rachel planted some flowers. Sam gave a watering can to {{Rachel}}  
indirect object: Rachel  

Number of important tokens: 1  
9_test: During the game, <<Chris>> and Kyle passed the ball to each other. Chris threw the ball to {{Kyle}}  
indirect object: Kyle  

Number of important tokens: 2  
10_test: While chatting, <<Steven>> and <<Roman>> were at the cafe. Steven gave a drawing to {{Roman}}  
indirect object: Roman 
"""

is_valid = validate_highlighted_output(input)
if is_valid:
    print("All highlighted outputs are valid.")
else:
    print("There are invalid highlighted outputs.")
