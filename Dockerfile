
FROM shaunmulligan995/rpi5-ai-kit:4.18.1 

# Bring our source code into docker context, everything not in .dockerignore
COPY . . 

# launch our app.
RUN chmod +x start.sh
CMD ["./start.sh"]