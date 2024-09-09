import {
  END,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

const systemMessage = {
  role: 'system',
  content:
    `You are a translator. Your task is to translate messages between languages. Determine the source and target languages based on the human message or chat context, and maintain that translation direction. Only change the direction if explicitly asked. Provide only the translation without any additional commentary or explanations.`,
};

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const { messages } = state;

  // Remove the tool binding
  // const llmWithTools = llm.bindTools(tools);
  // Prepend the system message to the messages array
  const allMessages = [systemMessage, ...messages];
  const result = await llm.invoke(allMessages);
  return { messages: [result] };
};

const shouldContinue = (state: typeof MessagesAnnotation.State) => {
  // Always end after the LLM response
  return END;
};

/**
 * MessagesAnnotation is a pre-built state annotation imported from @langchain/langgraph.
 * It is the same as the following annotation:
 *
 * ```typescript
 * const MessagesAnnotation = Annotation.Root({
 *   messages: Annotation<BaseMessage[]>({
 *     reducer: messagesStateReducer,
 *     default: () => [systemMessage],
 *   }),
 * });
 * ```
 */
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, [END]);

export const graph = workflow.compile({
  // The LangGraph Studio/Cloud API will automatically add a checkpointer
  // only uncomment if running locally
  // checkpointer: new MemorySaver(),
});
