


def return_agent(args):
    if args.agent.startswith('template'):
        from agents.template_policy import TemplatePolicy
        return TemplatePolicy(args)
    elif args.agent.startswith('graph2edits'):
        from agents.graph2edits_policy import Graph2EditsPolicy
        return Graph2EditsPolicy(args)
    else:
        raise Exception('Invalid agent!')