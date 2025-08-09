from hg_chat import HGChat

# Initialize the chat model
bro_chat = HGChat("doublebank/bro-chatbot")

if __name__=='__main__':
    messages = []
    system_prompt = "You're the best bro, Andy."
    while True:
        print("-"*20, "USER", "-"*20)
        user_input = input("You: ",)
        if user_input.startswith("/exit"):
            break
        messages.append(bro_chat.UserMessage(text=user_input))
        # response = bro_chat.run(system_prompt, messages)
        print("-"*20, "ASSISTANT", "-"*20)
        response = bro_chat.stream(system_prompt, messages, return_full_text=True, max_new_tokens=1000)
        messages.append(bro_chat.AIMessage(text=response))
        # print(response)
    messages.append(bro_chat.UserMessage(text="I gotta go now see ya."))
    response = bro_chat.stream(system_prompt, messages, return_full_text=True)