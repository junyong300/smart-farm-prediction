from shiny import App, reactive, render, ui

app_ui = ui.page_fluid(
    ui.navset_bar(
        ui.nav_panel(
            "ESPD",
            ui.layout_column_wrap(
                ui.card(
                    ui.input_action_button("action_button", "Action"),
                    ui.output_text("counter"),
                ),
                ui.card(
                )
            ),
        ),
        ui.nav_panel("FRUTNET", "Page B content"),
        id="selected_navset_bar",
        title="Smart Farm Predction",
    )
)


def server(input, output, session):
    @render.text()
    @reactive.event(input.action_button)
    def counter():
        return f"{input.action_button()}"


"""
shiny run --host 0.0.0.0 --port 8002 --reload app.py
"""

app = App(app_ui, server)
