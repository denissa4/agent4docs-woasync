FROM node:16-alpine

ARG DataSource
ENV DataSource=${DataSource}
ARG DbUser
ENV DbUser=${DbUser}
ARG DbPassword
ENV DbPassword=${DbPassword}
ARG DbName
ENV DbName=${DbName}
ARG DbPort
ENV DbPort=${DbPort}
ARG ApiEndPoint
ENV ApiEndPoint=${ApiEndPoint}
ARG ApiToken
ENV ApiToken=${ApiToken}
ARG AppId
ENV AppId=${LuisAppId}
ARG AppPassword
ENV AppPassword=${AppPassword}
ARG StaticEndPoint
ENV StaticEndPoint=${StaticEndPoint}

WORKDIR /app
RUN apk add py3-pip
RUN apk add gcc musl-dev python3-dev libffi-dev openssl-dev cargo make
RUN pip install --upgrade pip
RUN pip install azure-cli

COPY . .
RUN npm install && npm cache clean --force
RUN npm run build

CMD ["npm", "start"]
