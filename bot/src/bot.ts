// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import * as path from "path";
import { config } from "dotenv";
const ENV_FILE = path.join(__dirname, "..", ".env");
config({ path: ENV_FILE });

import { Activity, ActivityHandler, ActivityTypes, CardAction, CardFactory, CardImage, MessageFactory, TurnContext } from "botbuilder";

import axios from "axios";

interface NLSQLAnswer {
    answer: string;
    answer_type: string;
    unaccounted: string;
    addition_buttons: CardAction[];
    buttons: CardAction[];
    images: CardImage[];
    card_data: any;
}

interface botOptions {
    nlApiUrl: string;
    debug: boolean;
}

export class Bot extends ActivityHandler {
    debug: boolean;
    nlApiURL: string;

    constructor(botOptions: botOptions) {
        super();

        this.debug = botOptions.debug;
        this.nlApiURL = botOptions.nlApiUrl;

        if (this.debug) console.log("botOptions", botOptions);

        // See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
        this.onMessage(async (context, next) => {
            if (this.debug) console.log("message start");

            await Bot.createActivityTyping(context);

            let nlsql_answer: NLSQLAnswer;
            // # nlsql logic and parsing answer
            if (process.env.TalkToDocs === "True") {
                nlsql_answer = await this.talkToDocsPost(context.activity.channelId, context.activity.text);
            } else {
                nlsql_answer = await this.apiPost(context.activity.channelId, context.activity.text);
            }

            if (this.debug) console.log("nlsql_answer: ", nlsql_answer);

            switch (nlsql_answer["answer_type"]) {
                case "text":
                    await this.textAnswer(context, nlsql_answer["answer"]);
                    break;
                case "hero_card":
                    await this.heroCardAnswer(context, nlsql_answer["answer"], nlsql_answer["buttons"], nlsql_answer["images"]);
                    break;
                case "adaptive_card":
                    await this.adaptiveCardAnswer(context, nlsql_answer["card_data"]);
                    break;
                default:
                    throw new Error("NotImplemented");
            }

            if (nlsql_answer["unaccounted"] != null) {
                await this.textAnswer(context, nlsql_answer["unaccounted"]);
            }

            if (nlsql_answer["addition_buttons"] != null) {
                await this.heroCardAnswer(context, "", nlsql_answer["addition_buttons"], null);
            }

            // By calling next() you ensure that the next BotHandler is run.
            await next();
        });

        this.onMembersAdded(async (context, next) => {
            const membersAdded = context.activity.membersAdded;
            const welcomeText = "Hello and welcome!";
            for (const member of membersAdded) {
                if (member.id !== context.activity.recipient.id) {
                    await context.sendActivity(MessageFactory.text(welcomeText, welcomeText));
                }
            }
            // By calling next() you ensure that the next BotHandler is run.
            await next();
        });
    }

    private static async createActivityTyping(context) {
        let activity: Partial<Activity> = {
            type: ActivityTypes.Typing,
            channelId: context.activity.channelId,
            conversation: context.activity.conversation,
            recipient: context.activity.from,
            from: context.activity.recipient,
            attachmentLayout: "carousel",
            text: "",
            serviceUrl: context.activity.serviceUrl,
        };

        return await context.sendActivity(activity);
    }

    private async heroCardAnswer(context: TurnContext, text: string, buttons: any, images: any) {
        if (this.debug) console.log("heroCardAnswer");

        let imagesList: CardImage[] = [];
        if (images) {
            for (const img of images) {
                let cardImage: CardImage = {
                    url: img["img_url"],
                };
                imagesList.push(cardImage);
            }
        }

        let buttonsList: CardAction[] = [];
        if (buttons) {
            for (const btn of buttons) {
                let cardAction: CardAction = {
                    type: btn["type"],
                    title: btn["title"],
                    value: btn["value"],
                };
                buttonsList.push(cardAction);
            }
        }

        const attachment = CardFactory.heroCard("", text, imagesList, buttonsList);
        const response = MessageFactory.attachment(attachment, "");

        return await context.sendActivity(response);
    }

    private async textAnswer(context: TurnContext, text: string) {
        if (this.debug) console.log("textAnswer");

        let activity: Partial<Activity> = {
            type: ActivityTypes.Message,
            channelId: context.activity.channelId,
            conversation: context.activity.conversation,
            recipient: context.activity.from,
            from: context.activity.recipient,
            attachmentLayout: "carousel",
            text: text,
            serviceUrl: context.activity.serviceUrl,
        };

        // TODO there is no attachment in function args
        // if (attachment) {
        //     activity.attachments = attachment
        // }

        return await context.sendActivity(activity);
    }

    private async adaptiveCardAnswer(context: TurnContext, cardData: any) {
        if (this.debug) console.log("AdaptiveCardAnswer");

        const attachments = CardFactory.adaptiveCard(cardData);
        const response = MessageFactory.carousel([attachments], "");

        return await context.sendActivity(response);
    }

    // TODO Pass URL variable in here
    private async apiPost(channelId: string, text: string) {
        const body = {
            channel_id: channelId,
            text: text,
        };

        if (this.debug) console.log("requesting API");

        const response = await axios.post(this.nlApiURL, body);

        if (this.debug) console.log("response: ", response.data);

        return response.data;
    }

    private async talkToDocsPost(channelId: string, text: string) {
        // Construct the request body
        const body = {
            channel_id: channelId,
            text: text,
        };

        if (this.debug) console.log("Sending request to Flask server:", body);

        // Flask server query endpoint
        const flaskServerUrl = process.env.AzureAppName;

        // Send the request to the Flask server
        const response = await axios.post(`${flaskServerUrl}/query`, body, { timeout: 240000 });

        if (this.debug) console.log("Received response from Flask server:", response.data);

        // Return the API response from the Flask server
        console.log(response.data);

        const nlsqlAnswer: NLSQLAnswer = response.data;

        return nlsqlAnswer;
    }
}
