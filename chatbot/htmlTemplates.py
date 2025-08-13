css = r'''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    align-items: center;
}
.chat-message.user {
    background-color: #2b313e; */this is a dark blue color*/
}
.chat-message.bot {
    background-color: #475063; */this is a light blue color*/
}
.chat-message .avatar {
    width: 50px;
    height: 50px;
    min-width: 50px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 1rem;
}
.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.chat-message .message {
    flex: 1;
    color: #fff; */white text color*/
    word-wrap: break-word;
}
</style>
'''

bot_template = r'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="./images/sheru2.png" alt="AI Assistant">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = r'''
<div class="chat-message user">
    <div class="avatar">
        <img src="./images/sathisheru2.png" alt="User">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''