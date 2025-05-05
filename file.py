import streamlit as st
import graphviz
from collections import deque, defaultdict
import string

class State:
    def __init__(self, name):
        self.name = name
        self.transitions = defaultdict(list)
        self.is_final = False
        self.id = id(self)
        self.epsilon_transitions = []
       
class NFA:
    def __init__(self):
        self.states = {}
        self.start_state = None
        self.alphabet = set()

    def add_state(self, name):
        if name not in self.states:
            self.states[name] = State(name)
        return self.states[name]

    def add_transition(self, from_state, to_state, symbol):
        self.states[from_state].transitions[symbol].append(to_state)
        if symbol != 'ε':
            self.alphabet.add(symbol)

    def epsilon_closure(self, states):
        closure = set(states)
        queue = deque(states)
        
        while queue:
            current = queue.popleft()
            for state in self.states[current].transitions.get('ε', []):
                if state not in closure:
                    closure.add(state)
                    queue.append(state)
        return frozenset(closure)

    def move(self, states, symbol):
        result = set()
        for state in states:
            result.update(self.states[state].transitions.get(symbol, []))
        return result

    def to_dfa(self):
        dfa = DFA()
        initial_closure = self.epsilon_closure({self.start_state})
        
        # Système de nommage A, B, C...
        letter_index = 0
        state_mapping = {}
        state_mapping[frozenset(initial_closure)] = 'A'
        
        dfa.start_state = 'A'
        dfa.add_state('A')
        dfa.state_compositions['A'] = initial_closure
        
        if any(self.states[s].is_final for s in initial_closure):
            dfa.states['A'].is_final = True
        
        unmarked = [initial_closure]
        marked = []
        
        while unmarked:
            current = unmarked.pop()
            marked.append(current)
            current_id = state_mapping[frozenset(current)]
            
            for symbol in self.alphabet:
                next_states = self.epsilon_closure(self.move(current, symbol))
                if not next_states:
                    continue
                
                if frozenset(next_states) not in state_mapping:
                    letter_index += 1
                    new_id = string.ascii_uppercase[letter_index]
                    state_mapping[frozenset(next_states)] = new_id
                    dfa.add_state(new_id)
                    dfa.state_compositions[new_id] = next_states
                    
                    if any(self.states[s].is_final for s in next_states):
                        dfa.states[new_id].is_final = True
                    unmarked.append(next_states)
                
                next_id = state_mapping[frozenset(next_states)]
                dfa.add_transition(current_id, next_id, symbol)
        
        return dfa

class DFA:
    def __init__(self):
        self.states = {}
        self.start_state = None
        self.alphabet = set()
        self.state_compositions = {}  # Stocke la composition des états

    def add_state(self, name):
        if name not in self.states:
            self.states[name] = State(name)
        return self.states[name]

    def add_transition(self, from_state, to_state, symbol):
        self.states[from_state].transitions[symbol] = to_state
        self.alphabet.add(symbol)

def visualize_automaton(automaton, title):
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', labelloc='t', label=title)
    
    # Configuration des nœuds
    graph.attr('node', shape='circle', width='0.6', height='0.6')
    
    # Pour DFA, on utilise juste les lettres
    if isinstance(automaton, DFA):
        for state_id, state in automaton.states.items():
            if state.is_final:
                graph.node(state_id, shape='doublecircle')
            else:
                graph.node(state_id)
    else:
        # Pour NFA, on garde les noms originaux
        for state_name, state in automaton.states.items():
            if state.is_final:
                graph.node(state_name, shape='doublecircle')
            else:
                graph.node(state_name)
    
    # Flèche de départ
    graph.node('start', shape='none', label='')
    graph.edge('start', automaton.start_state)
    
    # Transitions
    for state_name, state in automaton.states.items():
        for symbol, targets in state.transitions.items():
            if isinstance(automaton, DFA):
                graph.edge(state_name, targets, label=symbol)
            else:
                for target in targets:
                    graph.edge(state_name, target, label=symbol)
    
    st.graphviz_chart(graph, use_container_width=True)
def regex_to_postfix(regex):
    precedence = {'*': 4, '+': 4, '?': 4, '.': 3, '|': 2}
    output = []
    stack = []
    new_regex = ""

    # Ajouter les concaténations explicites
    for i in range(len(regex)):
        c1 = regex[i]
        new_regex += c1
        if i + 1 < len(regex):
            c2 = regex[i+1]
            if (c1 not in '(|' and c2 not in '|)*+?'):
                new_regex += '.'

    # Conversion en postfixe (Shunting Yard)
    for char in new_regex:
        if char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif char in precedence:
            while (stack and stack[-1] != '(' and 
                   precedence[stack[-1]] >= precedence[char]):
                output.append(stack.pop())
            stack.append(char)
        else:
            output.append(char)

    while stack:
        output.append(stack.pop())

    return ''.join(output)




def regex_to_nfa(postfix):
    if not postfix:
        return NFA()

    stack = []
    state_count = 0  # Pour générer des noms uniques aux états

    def new_state(is_final=False):
        nonlocal state_count
        state = State(name=f"q{state_count}")
        state.is_final = is_final
        state_count += 1
        return state

    for token in postfix:
       
        if token == '.':
         
            nfa2 = stack.pop()
            nfa1 = stack.pop()

            # Ajout de la transition epsilon
            nfa1.states[nfa1.end_state].epsilon_transitions.append(nfa2.start_state)
            nfa1.states[nfa1.end_state].is_final = False

            # Fusionner les états
            combined_nfa = NFA()
            combined_nfa.states = nfa1.states.copy()
            combined_nfa.states.update(nfa2.states)  # Fusionner les deux dictionnaires
            combined_nfa.alphabet = nfa1.alphabet.union(nfa2.alphabet)
            combined_nfa.start_state = nfa1.start_state
            combined_nfa.end_state = nfa2.end_state
            stack.append(combined_nfa)

        elif token == '|':
          
            nfa2 = stack.pop()
            nfa1 = stack.pop()

            start = new_state()
            end = new_state(is_final=True)

            alt_nfa = NFA()

            # Copier les états des automates nfa1 et nfa2
            alt_nfa.states = {state.name: state for state in nfa1.states.values()}
            alt_nfa.states.update({state.name: state for state in nfa2.states.values()})

            # Ajouter les états start et end
            alt_nfa.states[start.name] = start
            alt_nfa.states[end.name] = end

            # Ajouter les transitions epsilon
            start.epsilon_transitions.extend([nfa1.start_state, nfa2.start_state])  # Transitions epsilon du start
            alt_nfa.states[nfa1.end_state].epsilon_transitions.append(end.name)  # Transition epsilon vers l'état final
            alt_nfa.states[nfa2.end_state].epsilon_transitions.append(end.name)  # Transition epsilon vers l'état final

            # Mettre à jour les états finaux
            alt_nfa.states[nfa1.end_state].is_final = False
            alt_nfa.states[nfa2.end_state].is_final = False

            alt_nfa.alphabet = nfa1.alphabet.union(nfa2.alphabet)

            # Définir l'état de départ et final
            alt_nfa.start_state = start.name
            alt_nfa.end_state = end.name

            stack.append(alt_nfa)

        elif token == '*':
           
            nfa = stack.pop()
            start = new_state()
            end = new_state(is_final=True)

            kleene_nfa = NFA()
            kleene_nfa.start_state = start.name
            kleene_nfa.end_state = end.name

            kleene_nfa.states = {start.name: start, end.name: end}
            kleene_nfa.states.update(nfa.states)  # Ajouter les états de nfa à kleene_nfa

            # Ajouter les transitions epsilon pour Kleene star
            start.epsilon_transitions.extend([nfa.start_state, end.name])
            kleene_nfa.states[nfa.end_state].epsilon_transitions.extend([nfa.start_state, end.name])
            kleene_nfa.states[nfa.end_state].is_final = False

            kleene_nfa.alphabet = nfa.alphabet
            kleene_nfa.start_state = start.name
            kleene_nfa.end_state = end.name

            stack.append(kleene_nfa)

        else:
          
            start = new_state()
            end = new_state(is_final=True)
            basic_nfa = NFA()
            basic_nfa.start_state = start.name
            basic_nfa.end_state = end.name
            if token not in start.transitions:
                start.transitions[token] = []
            start.transitions[token].append(end.name)
            basic_nfa.states[start.name] = start
            basic_nfa.states[end.name] = end
            basic_nfa.alphabet.add(token)
            stack.append(basic_nfa)

    result = stack.pop()
  
    return result



def convert_nfa_for_display(nfa):
    # Fonction pour reconstruire un NFA pour affichage et conversion
    compat_nfa = NFA()

   
    for state in nfa.states.values():
        
        compat_nfa.add_state(state.name)
       
        if state.is_final:
           
            compat_nfa.states[state.name].is_final = True

    # Ajout des transitions
    
    for state in nfa.states.values():
        for sym, targets in state.transitions.items():
            for target in targets:
               
                target_state = nfa.states.get(target)
                compat_nfa.add_transition(state.name, target_state.name, sym)
        for target in state.epsilon_transitions:
            target_state = nfa.states.get(target)
            compat_nfa.add_transition(state.name, target_state.name, 'ε')
   
    # Définir l'état initial par nom
    compat_nfa.start_state = nfa.start_state

    return compat_nfa
# Streamlit app
def main():
    st.title("Projet Module IA2 - Automates Finis")
    st.subheader("Construction et déterminisation d'ε-NFA")
    
    tab1, tab2 = st.tabs(["Construction ε-NFA", "Déterminisation en AFD"])
    
    with tab1:
        st.header("Construction d'un ε-NFA")
        
        construction_method = st.radio("Méthode de construction", 
                                    ["Manuelle", "Depuis une expression régulière"])
        
        if construction_method == "Manuelle":
            col1, col2 = st.columns(2)
            
            with col1:
                num_states = st.number_input("Nombre d'états", min_value=1, value=3, step=1)
                alphabet = st.text_input("Alphabet (séparé par des virgules)", value="a,b")
                alphabet = [x.strip() for x in alphabet.split(',') if x.strip()]
                alphabet.append('ε')  # Always include epsilon
                
                states = [f"q{i}" for i in range(num_states)]
                start_state = st.selectbox("État initial", states)
                final_states = st.multiselect("États finaux", states)
                
                st.markdown("**Transitions**")
                transitions = []
                
                for i in range(st.number_input("Nombre de transitions", min_value=0, value=3, step=1)):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        from_state = st.selectbox(f"De", states, key=f"from_{i}")
                    with col_b:
                        symbol = st.selectbox(f"Symbole", alphabet, key=f"sym_{i}")
                    with col_c:
                        to_state = st.selectbox(f"À", states, key=f"to_{i}")
                    transitions.append((from_state, symbol, to_state))
            
            with col2:
                if st.button("Construire ε-NFA (Manuel)"):
                    nfa = NFA()
                    
                    for state in states:
                        nfa.add_state(state)
                    
                    nfa.start_state = start_state
                    
                    for state in final_states:
                        nfa.states[state].is_final = True
                    
                    for from_state, symbol, to_state in transitions:
                        nfa.add_transition(from_state, to_state, symbol)
                    
                    st.session_state.nfa = nfa
                    st.success("ε-NFA construit avec succès!")
                    
                    st.subheader("Visualisation de l'ε-NFA")
                    visualize_automaton(nfa, "ε-NFA")
                    
                    # Display transition table
                    st.subheader("Table de transitions ε-NFA")
                    trans_table = []
                    for state in states:
                        for symbol in alphabet:
                            if symbol in nfa.states[state].transitions:
                                trans_table.append({
                                    "État": state,
                                    "Symbole": symbol,
                                    "Vers": ', '.join(nfa.states[state].transitions[symbol])
                                })
                    
                    st.table(trans_table)
        
        else:  # Construction depuis expression régulière
            st.subheader("Construction depuis une expression régulière")
            regex = st.text_input("Entrez une expression régulière (ex: a(b|c)*)", "a(b|c)*")
            
            if st.button("Construire ε-NFA (Expression)"):
                try:
                    postfix = regex_to_postfix(regex)
                    st.write(f"Expression postfixée: {postfix}")
                    nfa = regex_to_nfa(postfix)
                    

                    compat_nfa = convert_nfa_for_display(nfa)
                   

                    # Sauvegarde et affichage
                    st.session_state.nfa = compat_nfa
                    st.success("ε-NFA construit avec succès!")

                    # Visualisation de l'automate
                    visualize_automaton(compat_nfa, f"ε-NFA pour {regex}")

                    # Affichage de la table des transitions
                    st.subheader("Table de transitions ε-NFA")
                    trans_table = []
                    for state_name, state in compat_nfa.states.items():
                        for symbol in sorted(state.transitions.keys()):
                            trans_table.append({
                                "État": state_name,
                                "Symbole": symbol,
                                "Vers": ', '.join(state.transitions[symbol])
                            })
                        if state.epsilon_transitions:
                            trans_table.append({
                                "État": state_name,
                                "Symbole": 'ε',
                                "Vers": ', '.join(state.epsilon_transitions)
                            })

                    st.table(trans_table)
                    
                except Exception as e:
                    st.error(f"Erreur dans la construction: {str(e)}")

    with tab2:
        st.header("Déterminisation en AFD")
        
        if 'nfa' not in st.session_state:
            st.warning("Veuillez d'abord construire un ε-NFA dans l'onglet précédent.")
        else:
            nfa = st.session_state.nfa
            
            if st.button("Convertir en AFD"):
                dfa = nfa.to_dfa()
                st.session_state.dfa = dfa
                st.success("AFD généré avec succès!")
                
                st.subheader("Visualisation de l'AFD")
                visualize_automaton(dfa, "AFD")
                
                # Display DFA transition table
                st.subheader("Table de transitions AFD")
                trans_table = []
                for from_state, state in dfa.states.items():
                    from_comp = ",".join(sorted(dfa.state_compositions[from_state]))
                    for symbol in dfa.alphabet:
                        if symbol in state.transitions:
                            to_state = state.transitions[symbol]
                            to_comp = ",".join(sorted(dfa.state_compositions[to_state]))
                            trans_table.append({
                                "État": f"{from_state}={{{from_comp}}}",
                                "Symbole": symbol,
                                "Vers": f"{to_state}={{{to_comp}}}"
                            })
                
                st.table(trans_table)
                
                # Comparison
                st.subheader("Comparaison ε-NFA/AFD")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nombre d'états ε-NFA", len(nfa.states))
                with col2:
                    st.metric("Nombre d'états AFD", len(dfa.states))

if __name__ == "__main__":
    main()